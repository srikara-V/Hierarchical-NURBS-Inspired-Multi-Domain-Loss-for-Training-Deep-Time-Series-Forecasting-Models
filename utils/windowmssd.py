import torch

class WindowedAverageCache:
    def __init__(self):
        self.cache = {}
    
    def get_windowed_average_basis(self, x, window_size: int):
        # Create a cache key using relevant parameters
        cache_key = (window_size, x.device.type)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            basis = self._compute_windowed_average_basis(x, window_size)
            self.cache[cache_key] = basis
            return basis

    @staticmethod
    def _compute_windowed_average_basis(x, window_size: int):
        n_points = len(x)
        basis = torch.zeros((n_points - window_size + 1, n_points), device=x.device)
        
        for i in range(n_points - window_size + 1):
            basis[i, i:i + window_size] = 1.0 / window_size
        
        return basis

def windowed_average_decompose_time_series(time_series, cache: WindowedAverageCache, max_levels: int=5, window_size: int=10, threshold: float=1.25e-50, scaling_factor: int=1):
    seq_length = time_series.shape[1]
    batch_size, _, num_channels = time_series.shape
    device = time_series.device
    
    decomposed = []
    residual = time_series.clone()
    
    for level in range(max_levels):
        # Ensure the window size is within valid range
        current_window_size = min(window_size * (2 ** level), seq_length)
        if current_window_size >= seq_length:
            break
        
        x = torch.arange(seq_length, dtype=torch.float32, device=device)
        
        # Retrieve the cached basis or compute it if not available
        basis = cache.get_windowed_average_basis(x, current_window_size)
        
        residual_flat = residual.reshape(-1, seq_length).t()
        coefs = torch.matmul(basis, residual_flat)
        
        current_level = torch.matmul(basis.t(), coefs).t().reshape(batch_size, seq_length, num_channels)
        
        decomposed.append(current_level)
        residual -= current_level
        residual *= scaling_factor

        if torch.max(torch.std(residual, dim=1)) < threshold * torch.max(torch.std(time_series, dim=1)):
            break

    decomposed.append(residual)

    return torch.stack(decomposed, dim=-1)

class WindowedAverageDecomposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cache, y_true_decomposed, max_levels, exponent=5, window_size=10, threshold=1.25e-50, scaling_factor=1):
        
        decomposed = windowed_average_decompose_time_series(tensor, cache, max_levels, window_size=window_size, threshold=threshold, scaling_factor=scaling_factor)
        
        level_losses = torch.mean(torch.abs(decomposed - y_true_decomposed)**exponent, dim=tuple(range(decomposed.ndim - 1)))
        
        total_loss = torch.sum(level_losses)
        loss_weights = level_losses / (total_loss + 1e-8)
        
        ctx.save_for_backward(tensor, loss_weights)
        
        return decomposed

    @staticmethod
    def backward(ctx, grad_output):
        tensor, loss_weights = ctx.saved_tensors
        
        grad_input = grad_output * loss_weights.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        grad_input = grad_input.sum(dim=-1)
        
        return grad_input, None, None, None, None, None

def decompose_and_reconstruct_windowed_average(tensor, cache, y_true, max_levels=5, exponent=5, window_size=10, threshold=1.25e-50, scaling_factor=1):

    return WindowedAverageDecomposition.apply(tensor, cache, y_true, max_levels, exponent, window_size, threshold, scaling_factor)

import torch
import torch.nn as nn

class MSSD(nn.Module):
    def __init__(self, num_variables, sequence_length, max_levels=5, alpha=.1, beta=1, gamma=1, window_size=10, threshold=1e-50, residual_scaling_factor=1, exponent=5):
        super(MSSD, self).__init__()
        self.num_variables = num_variables
        self.sequence_length = sequence_length
        self.max_sum = sequence_length * 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_levels = max_levels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.threshold = threshold
        self.residual_scaling_factor = residual_scaling_factor
        self.cache = WindowedAverageCache()
        self.exponent = exponent
    
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val + 1e-7)
    
    def compute_losses(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        deconstructed_true = windowed_average_decompose_time_series(
            y_true, self.cache, self.max_levels, window_size=self.window_size, 
            threshold=self.threshold, scaling_factor=self.residual_scaling_factor
        )
        
        deconstructed_pred = windowed_average_decompose_time_series(
            y_pred, self.cache, self.max_levels, window_size=self.window_size, 
            threshold=self.threshold, scaling_factor=self.residual_scaling_factor
        )

        mae_loss = torch.abs(
            self.normalize(deconstructed_pred, self.norm_params["min_value"], self.norm_params["max_value"]) - 
            self.normalize(deconstructed_true, self.norm_params["min_value"], self.norm_params["max_value"])
        )

        diff_pred = torch.diff(deconstructed_pred, dim=1)
        diff_true = torch.diff(deconstructed_true, dim=1)
        diff_loss = torch.abs(self.normalize(diff_pred, self.norm_params["min_slope"], self.norm_params["max_slope"]) - 
                              self.normalize(diff_true, self.norm_params["min_slope"], self.norm_params["max_slope"]))
        
        fft_pred = torch.fft.rfft(deconstructed_pred, dim=1)
        fft_true = torch.fft.rfft(deconstructed_true, dim=1)
        fft_mae = torch.abs(self.normalize(fft_pred, self.norm_params["min_fft"], self.norm_params["max_fft"]) - 
                            self.normalize(fft_true, self.norm_params["min_fft"], self.norm_params["max_fft"]))
        
        return mae_loss, diff_loss, fft_mae
    
    def forward(self, y_pred, y_true):
        mae_loss, diff_loss, fft_mae = self.compute_losses(y_pred, y_true)

        total_loss = self.alpha * mae_loss.mean() + self.beta * diff_loss.mean() + self.gamma * fft_mae.mean() 

        return total_loss

    def set_norm_params(self, dataloaders):
        self.norm_params = self.analyze_time_series(dataloaders)
    
    def analyze_time_series(self, dataloaders):
        global_max = -float('inf')
        global_min = float('inf')
        global_max_slope = -float('inf')
        global_min_slope = float('inf')
        global_max_diff_2 = -float('inf')
        global_min_diff_2 = float('inf')
        global_max_fft = -float('inf')
        global_min_fft = float('inf')
        
        for dataloader in dataloaders:
            for batch in dataloader:
                data = batch[0]  # Extract the data tensor from the batch
                
                # Calculate max and min values
                batch_max = data.max().item()
                batch_min = data.min().item()
                
                global_max = max(global_max, batch_max)
                global_min = min(global_min, batch_min)
                
                # Calculate slopes
                slopes = torch.diff(data, dim=1)
                
                # Calculate max and min slopes
                max_slope = slopes.max().item()
                min_slope = slopes.min().item()
                
                global_max_slope = max(global_max_slope, max_slope)
                global_min_slope = min(global_min_slope, min_slope)

                diff_2s = torch.diff(slopes, dim=1)
                
                # Calculate max and min diff_2
                max_diff_2 = diff_2s.max().item()
                min_diff_2 = diff_2s.min().item()
                
                global_max_diff_2 = max(global_max_diff_2, max_diff_2)
                global_min_diff_2 = min(global_min_diff_2, min_diff_2)

                # Calculate FFT
                fft_values = torch.abs(torch.fft.rfft(data, dim=1))
                max_fft = fft_values.max().item()
                min_fft = fft_values.min().item()

                global_max_fft = max(global_max_fft, max_fft)
                global_min_fft = min(global_min_fft, min_fft)
        
        return {
            "max_value": global_max,
            "min_value": global_min,
            "max_slope": global_max_slope,
            "min_slope": global_min_slope,
            "max_diff_2": global_max_diff_2,
            "min_diff_2": global_min_diff_2,
            "max_fft": global_max_fft,
            "min_fft": global_min_fft
        }
