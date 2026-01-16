import itertools
import os
import random
import torch
import numpy as np

# Define the lists of parameters
root_path_list = ['traffic']
data_path_list = ['traffic.csv']
data_list = ['custom']
pred_len_list = [96, 192, 336, 720]
model_name_list = ['MLP']
loss_function_list = ['mssd']
exponents = [1,2,5,8]
knot_scalings = [1,2,3,5]

# Iterate through each combination
for model_name in model_name_list:
  for root_path, data_path, data in zip(root_path_list, data_path_list, data_list):
      for pred_len in pred_len_list:
            fix_seed = random.randint(1, 10000)
            random.seed(fix_seed)
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            for loss_function in loss_function_list:
              if loss_function == 'mssd':
                for knot_mult in knot_scalings:
                   for exponent in exponents:
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
                        f"--knot_multiplier {knot_mult} "
                        f"--spline_criterion_exponent {exponent}"
                    )
                    print(f"Executing command: {command}")
                    os.system(command)

