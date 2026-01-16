import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft as fft

def get_nurbs_weights(time_series, window_size):
    if window_size == 0:
        window_size = 4
    pad = window_size // 2
    padded = F.pad(time_series, (0, 0, pad, pad), mode='replicate')
    variance = torch.var(padded.unfold(1, window_size, 1), dim=2)
    return F.softmax(variance, dim=1)

class NURBSBasisCache:
    def __init__(self):
        self.cache = {}
    
    def get_nurbs_basis(self, x, knots, weights, degree: int):
        cache_key = (degree, len(knots), knots.device.type, weights.sum().item())
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            basis = self._compute_nurbs_basis(x, knots, weights, degree)
            self.cache[cache_key] = basis
            return basis

    @staticmethod
    @torch.jit.script
    def _compute_nurbs_basis(x, knots, weights, degree: int = 3):
        n_knots = len(knots)
        n_splines = n_knots - degree - 1
        n_points = len(x)
        
        basis = torch.ones((n_splines, n_points), device=x.device)
        
        for d in range(1, degree + 1):
            left_knots = knots[:-d]
            right_knots = knots[d:]
            denom = right_knots - left_knots
            
            mask = (x.unsqueeze(1) >= left_knots.unsqueeze(0)) & (x.unsqueeze(1) < right_knots.unsqueeze(0))
            w = torch.where(mask, (x.unsqueeze(1) - left_knots.unsqueeze(0)) / (denom.unsqueeze(0) + 1e-8), torch.zeros_like(x).unsqueeze(1))
            
            basis_next = torch.zeros((n_splines + 1, n_points), device=x.device)
            basis_next[:-1] += (1 - w.t()[:n_splines]) * basis
            basis_next[1:] += w.t()[:n_splines] * basis
            
            basis = basis_next[:n_splines]
        
        # Reshape weights to match basis
        weights_reshaped = F.interpolate(weights.unsqueeze(0).unsqueeze(0), size=(n_splines, n_points), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        # Apply weights to convert B-spline basis to NURBS basis
        weighted_basis = basis * weights_reshaped
        nurbs_basis = weighted_basis / weighted_basis.sum(dim=0, keepdim=True)
        
        return nurbs_basis

def nurbs_decompose_time_series(time_series, cache, max_levels: int=5, degree: int=3, threshold: float=1.25e-50, scaling_factor: float=1.0, knot_scaling_factor: int=3):
    seq_length = time_series.shape[1]
    batch_size, _, num_channels = time_series.shape
    device = time_series.device
    
    decomposed = []
    residual = time_series.clone()
    
    for level in range(max_levels):
        num_knots = int(max(2*degree + 2, min(seq_length // 2, 2 ** (level + 2) + seq_length/6)) * knot_scaling_factor)
        knots = torch.linspace(0, seq_length, num_knots, device=device)
        x = torch.arange(seq_length, dtype=torch.float32, device=device)
        
        # Compute autocorrelation-based weights
        window_size = seq_length // num_knots
        weights = get_nurbs_weights(residual, window_size)
        
        # Use weights for the first channel
        basis = cache.get_nurbs_basis(x, knots, weights[:, :, 0], degree)
        
        residual_flat = residual.reshape(-1, seq_length).t()
        reg_term = torch.eye(basis.shape[0], device=device) * 1e-6
        basis_reg = torch.matmul(basis, basis.t()) + reg_term
        coefs = torch.linalg.solve(basis_reg, torch.matmul(basis, residual_flat))
        
        if torch.isnan(coefs).any() or torch.isinf(coefs).any():
            print(f"NaN or Inf coefficients encountered at level {level}. Stopping decomposition.")
            break
        
        coefs = coefs[:basis.shape[0]]
        current_level = torch.matmul(basis.t(), coefs).t().reshape(batch_size, seq_length, num_channels)
        
        decomposed.append(current_level)
        residual -= current_level
        residual *= scaling_factor

    decomposed_tensor = torch.stack(decomposed, dim=-1)
    
    # Compute products of adjacent level pairs
    level_products = []
    for i in range(decomposed_tensor.shape[-1]-1):
        level_products.append(decomposed_tensor[..., i] + decomposed_tensor[..., i+1])
    
    level_products_tensor = torch.stack(level_products, dim=-1)
    final_decomposition = torch.cat([decomposed_tensor, level_products_tensor], dim=-1)

    return final_decomposition

class NurbsDecomposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cache, y_true_decomposed, max_levels, exponent=5, degree=3, threshold=1.25e-50, scaling_factor=1, knot_scaling_factor=3):
        
        decomposed = nurbs_decompose_time_series(tensor, cache,max_levels, degree=degree, threshold=threshold, scaling_factor=scaling_factor, knot_scaling_factor=knot_scaling_factor)
        
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
        
        return grad_input, None, None, None, None, None, None, None, None
    
def decompose_and_reconstruct(tensor,cache, y_true, max_levels=5, exponent=5, degree=3, threshold=1.25e-50, scaling_factor=1, knot_scaling_factor=3):

    return NurbsDecomposition.apply(tensor, cache, y_true, max_levels, exponent, degree, threshold, scaling_factor, knot_scaling_factor)

class MSSD(nn.Module):
    def __init__(self, num_variables, sequence_length, max_levels=5, alpha=.1, beta=1, gamma=1, degree=3, threshold=1e-50, residual_scaling_factor=1, knot_scaling_factor=3, exponent=5):
        super(MSSD, self).__init__()
        self.num_variables = num_variables
        self.sequence_length = sequence_length
        self.max_sum = sequence_length * 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_levels = max_levels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.degree = degree
        self.threshold = threshold
        self.residual_scaling_factor = residual_scaling_factor
        self.knot_scaling_factor = knot_scaling_factor
        self.cache = NURBSBasisCache()
        self.spline_criterion_exponent = exponent
    
    
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val + 1e-7)
    
    def compute_losses(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        deconstructed_true = nurbs_decompose_time_series(y_true,self.cache, self.max_levels, degree=self.degree, threshold=self.threshold, scaling_factor=self.residual_scaling_factor, knot_scaling_factor=self.knot_scaling_factor)
        
        deconstructed_pred = decompose_and_reconstruct(y_pred, self.cache, deconstructed_true, self.max_levels, exponent=self.spline_criterion_exponent, degree=self.degree, threshold=self.threshold, scaling_factor=self.residual_scaling_factor, knot_scaling_factor=self.knot_scaling_factor)
        #deconstructed_true = decompose_and_reconstruct(y_true, y_true, self.max_levels)

        mae_loss = torch.abs(
            self.normalize(deconstructed_pred, self.norm_params["min_value"], self.norm_params["max_value"]) - 
            self.normalize(deconstructed_true, self.norm_params["min_value"], self.norm_params["max_value"]))

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
