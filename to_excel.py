import os
import numpy as np
import pandas as pd

# Set the base directory to the current directory
base_dir = os.getcwd()

# Initialize an empty list to store the data
data_list = []

# Iterate over each folder in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    if os.path.isdir(folder_path):
        # Parse the folder name using the given format
        setting_params = folder_name.split('_')[2:]

        # Initialize an empty list to store the combined parameters
        combined_params = []

        # Iterate over the setting_params using an index
        i = 0
        while i < len(setting_params):
            # Check if current and next element are 'exchange' and 'rate'
            if i + 1 < len(setting_params) and setting_params[i] == 'exchange' and setting_params[i + 1] == 'rate':
                combined_params.append('exchange_rate')
                i += 2  # Increment i by 2 to skip the 'rate' element
            else:
                combined_params.append(setting_params[i])
                i += 1

        setting_params = combined_params
        # Define the path to the metrics.npy file
        metrics_path = os.path.join(folder_path, 'metrics.npy')
        
        if os.path.exists(metrics_path):
            # Load the numpy array from the file
            metrics = np.load(metrics_path)
            
            # Extract values from the folder name
            setting = {
                'model_id': setting_params[0],
                'model': setting_params[1],
                'data': setting_params[2],
                'feature': setting_params[3],
                'data_path': setting_params[4],
                'seq_len': setting_params[5][2:],
                'label_len': setting_params[6][2:],
                'pred_len': setting_params[7][2:],
                'd_model': setting_params[8][2:],
                'n_heads': setting_params[9][2:],
                'e_layers': setting_params[10][2:],
                'd_layers': setting_params[11][2:],
                'd_ff': setting_params[12][2:],
                'factor': setting_params[13][2:],
                'embed': setting_params[14][2:],
                'distil': setting_params[15][2:],
                'des': setting_params[16][2:],
                'class_strategy': setting_params[17],
                'ii': setting_params[18][2:],
                'loss': setting_params[19][2:],
                'knot_multiplier': setting_params[20][2:],
                'spline_criterion_exponent': setting_params[21][2:],
                'alpha': setting_params[22],
                'bets': setting_params[23],
                'gamma': setting_params[24],

            }
            
            # Add metrics to the setting dictionary
            setting['metrics'] = metrics.tolist()
            
            # Append the setting dictionary to the data list
            data_list.append(setting)

# Convert the data list to a pandas DataFrame
df = pd.DataFrame(data_list)

# Save the DataFrame to an Excel file
output_path = os.path.join(base_dir, 'output.xlsx')
df.to_excel(output_path, index=False)

print(f'Data has been successfully saved to {output_path}')
