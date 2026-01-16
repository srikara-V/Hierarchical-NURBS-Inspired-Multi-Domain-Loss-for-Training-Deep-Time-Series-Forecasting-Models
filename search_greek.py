import itertools
import os
import random
import torch
import numpy as np

import pandas as pd


fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# Creating a dataframe from the provided data in the image
hyperparam_data = {
    "Dataset": [
        "ETTm1", "ETTm1", "ETTm1", "ETTm1",
        "ETTm2", "ETTm2", "ETTm2", "ETTm2",
        "ETTh1", "ETTh1", "ETTh1", "ETTh1",
        "ETTh2", "ETTh2", "ETTh2", "ETTh2",
        "electricity", "electricity", "electricity", "electricity",
        "exchange", "exchange", "exchange", "exchange",
        "traffic", "traffic", "traffic", "traffic",
        "weather", "weather", "weather", "weather"
    ],
    "Pred Length": [96, 192, 336, 720, 96, 192, 336, 720, 96, 192, 336, 720, 96, 192, 336, 720, 96, 192, 336, 720,
                    96, 192, 336, 720, 96, 192, 336, 720, 96, 192, 336, 720],
    "Max Levels": [5]*32,
    "Spline Degree": [3]*32,
    "Alpha": [.01,.01,.01,.01,  .1,.01,.1,.01,  .01,.01,.01,.01,  1.0,.1,.1,.1,  .01,.01,.01,.01,  .01,.01,.1,.1,  .01,.01,.01,.01,  .01,.01,.01,.01,],
    "Beta": [5,5,5,5,  1,1,1,1,  2,2,2,2,  0,1,0,2,  0,2,2,0,  1,0,0,0,  0,0,0,0,  1,0,0,0],
    "Gamma": [1,1,1,1,  1,1,5,5,  1,2,3,3, 1,1,0,3, 5,2,2,5,  1,3,2,3,  3,3,3,5,  1,5,3,5],
    "Knot Scaling Factor": [5,5,5,5,  3,5,5,5,  3,3,5,5,  5,5,5,5,  3,1,5,5,  5,5,3,4,  5,5,5,3,  3,5,5,5],
    "Spline Criterion Exponent": [1,1,1,1,  2,1,1,1,  1,1,1,1,  5,5,3,2,  2,1,3,1, 1,5,1,4,  5,5,5,1,  1,2,3,5],
    "Control Point Weight Window": [4]*32
}



param_df = pd.DataFrame(hyperparam_data)
param_df = pd.read_excel('HNMV_MLP_Hyperparam_testing_optimized.xlsx')
print(param_df.head())  # Display the dataframe

# Define the lists of parameters
root_path_list = ['ETT-small','ETT-small', 'electricity', 'exchange_rate', 'weather', 'traffic', 'ETT-small','ETT-small',]
data_path_list = ['ETTm1.csv', 'ETTm2.csv', 'electricity.csv', 'exchange_rate.csv', 'weather.csv', 'traffic.csv', 'ETTh1.csv', 'ETTh2.csv',]
data_list = ['ETTm1', 'ETTm2', 'custom', 'custom', 'custom', 'custom','ETTh1', 'ETTh2']
data_names = ['ETTm1', 'ETTm2','electricity', 'exchange_rate', 'weather', 'traffic','ETTh1', 'ETTh2']
pred_len_list = [96, 192, 336, 720]
model_name_list = ['MLP']
loss_function_list = ['mssd']



for model_name in model_name_list:
  for root_path, data_path, data, name in zip(root_path_list, data_path_list, data_list, data_names):
      for pred_len in pred_len_list:
            if name in ["ETTm1", "ETTm2", "electricity"]:
                continue
          
            selected_hp = param_df[(param_df['feature'] == name) & (param_df['pred_len'] == pred_len)]
            alpha = float(selected_hp.iloc[0]["alpha"][1:])
            beta = int(float(selected_hp.iloc[0]["bets"][1:]))
            gamma = int(float(selected_hp.iloc[0]["gamma"][1:]))
            ksf = float(selected_hp.iloc[0]["knot_multiplier"])
            sce = int(selected_hp.iloc[0]["spline_criterion_exponent"])

    
            for loss_function in loss_function_list:
                command = (
                        f"python -u run.py "
                        f"--is_training 1 "
                        f"--root_path ./data/{root_path}/ "
                        f"--data_path {data_path} "
                        f"--model_id {data}_{pred_len}_{model_name} "
                        f"--model {model_name} "
                        f"--data {data} "
                        f"--features M "
                        f"--seq_len 96 "
                        f"--pred_len {pred_len} "
                        f"--e_layers 2 "
                        f"--enc_in 7 "
                        f"--dec_in 7 "
                        f"--c_out 7 "
                        f"--des 'Exp' "
                        f"--d_model 256 "
                        f"--d_ff 512 "
                        f"--itr 1 "
                        f"--loss {loss_function} "
                        f"--knot_multiplier {ksf} "
                        f"--spline_criterion_exponent {sce} "
                        f"--alpha {alpha} "
                        f"--beta {beta} "
                        f"--gamma {gamma}"
                    )
                        
                print(f"Executing command: {command}")
                os.system(command)



                        

