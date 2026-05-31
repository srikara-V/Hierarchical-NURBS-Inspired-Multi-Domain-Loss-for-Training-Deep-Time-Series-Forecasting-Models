import argparse
import os
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from utils.config_defaults import apply_config_defaults
from utils.device import configure_training_device, empty_device_cache
from utils.experiment_paths import (
    build_setting_name,
    checkpoint_exists,
    metrics_exist,
    training_state_exists,
)
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results/', help='directory for metrics.npy outputs')
    parser.add_argument(
        '--skip_if_done',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='skip train/test when metrics.npy already exists for this setting',
    )
    parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='resume interrupted training from saved training_state.pth',
    )
    parser.add_argument(
        '--recover',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='if training finished but metrics are missing, run test only',
    )

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_core', type=int, default=512, help='dimension of core') # For SOFTS core
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--embed_type', type=int, default=0,
                        help='embedding type for Autoformer (0=wo_pos default)')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='use channel-independent linear layers for NLinear/Linear')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--seed', type=int, default=2024, help='random seed (incremented for each itr run)')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSSD', help='loss function')

    parser.add_argument('--knot_multiplier', type=float, default='3', help='loss function knot multiplier') # Multiplier on number of knots for MSSD
    parser.add_argument('--spline_criterion_exponent', type=int, default='5', help='loss function spline criterion exponent') 
    parser.add_argument('--alpha', type=float, default='.1', help='loss function spline criterion exponent') 
    parser.add_argument('--beta', type=float, default='1', help='loss function spline criterion exponent') 
    parser.add_argument('--gamma', type=float, default='1', help='loss function spline criterion exponent') 
    parser.add_argument('--max_levels', type=int, default='5', help='loss function spline criterion exponent') 



    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()
    args = apply_config_defaults(args)
    configure_training_device(args)

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            run_seed = args.seed + ii
            set_seed(run_seed)
            print(f'Using random seed: {run_seed}')

            setting = build_setting_name(args, ii)

            if args.skip_if_done and metrics_exist(setting, args.results_dir):
                print(f'>>>>>>>skip (already done): {setting}')
                continue

            exp = Exp(args)
            if (
                args.recover
                and checkpoint_exists(setting, args.checkpoints)
                and not training_state_exists(setting, args.checkpoints)
                and not metrics_exist(setting, args.results_dir)
            ):
                print(f'>>>>>>>recover (test only): {setting}')
                exp.test(setting, test=1)
            else:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting, resume=args.resume)

                if getattr(Exp, "_interrupt_requested", False):
                    print(f'Run interrupted; resume with the same command to continue: {setting}')
                    break

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            empty_device_cache(args.device)
    else:
        set_seed(args.seed)
        setting = build_setting_name(args, 0)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        empty_device_cache(args.device)
