import itertools
import os
import random
import torch
import numpy as np

# Define the lists of parameters
root_path_list = ['ETT-small','ETT-small', 'electricity', 'exchange_rate', 'weather', 'traffic', 'ETT-small','ETT-small',]
data_path_list = ['ETTm1.csv', 'ETTm2.csv', 'electricity.csv', 'exchange_rate.csv', 'weather.csv', 'traffic.csv', 'ETTh1.csv', 'ETTh2.csv',]
data_list = ['ETTm1', 'ETTm2', 'custom', 'custom', 'custom', 'custom','ETTh1', 'ETTh2']
pred_len_list = [96, 192, 336, 720]
model_name_list = ['MLP', 'DLinear', "SOFTS"]
loss_function_list = ['tildeq']

# Iterate through each combination
for model_name in model_name_list:
  for root_path, data_path, data in zip(root_path_list, data_path_list, data_list):
      for pred_len in pred_len_list:
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
                        f"--itr 3 "
                        f"--loss {loss_function} "
                        f"--knot_multiplier {0} "
                        f"--spline_criterion_exponent {0}"
                    )
                    print(f"Executing command: {command}")
                    os.system(command)

