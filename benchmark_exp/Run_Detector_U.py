import pandas as pd
import argparse, time, os

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import run_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

from benchmark_exp.configs import DEFAULT_PATHS_UNI as DEFAULT_PATHS
from benchmark_exp.utils import set_seed, print_cuda_info
from benchmark_exp.data_loader import load_file, train_split

seed = 2024
set_seed(seed)

print_cuda_info()

if __name__ == '__main__':

    Start_T = time.time()

    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_PATHS['dataset_dir'])
    parser.add_argument('--file_list', type=str, default=DEFAULT_PATHS['file_list'])
    parser.add_argument('--save_dir', type=str, default=DEFAULT_PATHS['save_dir'])
    parser.add_argument('--AD_Name', type=str, default='PretrainedForecaster')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]

    col_w = None 
    rows = []    
    done_files = set()

    save_path = f'{args.save_dir}/{args.AD_Name}.csv'
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        col_w = list(existing.columns)
        rows = existing.values.tolist()
        done_files = set(existing['file'].values)
        print(f'Found existing results for {len(done_files)} files in {save_path}')

    for i, filename in enumerate(file_list):

        if filename in done_files:
            print('Skipping (already done):{}'.format(filename))
            continue

        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        data, label, slidingWindow = load_file(file_path)
        data_train = train_split(data, filename)

        start_time = time.time()

        output = run_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
        run_time = time.time() - start_time

        try:
            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
            print('evaluation_result: ', evaluation_result)
            metrics = list(evaluation_result.values())
            if col_w is None:
                col_w = ['file', 'Time'] + list(evaluation_result.keys())
        except Exception:
            metrics = [0] * 9

        row = [filename, run_time] + metrics
        rows.append(row)

        if col_w is not None:
            pd.DataFrame(
                rows,
                columns=col_w,
            ).to_csv(save_path, index=False)