import os
import pandas as pd
from collections import namedtuple
import re
import csv
import sys
from configs import SmallConfig
import math

ModelPerformance = namedtuple('ModelPerformance', ['name', 'semcor', 'mun'])

def read_performance(base_dir, name=None):
    num_examples = 1644
    mun_path = os.path.join(base_dir, 'mun/results.txt')
    semcor_path = os.path.join(base_dir, 'semcor/results.txt')
    if os.path.exists(mun_path) and os.path.exists(semcor_path):
        with open(mun_path) as f:
            mun_correct = int(f.read())
        with open(semcor_path) as f:
            semcor_correct = int(f.read())
        return ModelPerformance(name, semcor_correct/num_examples, mun_correct/num_examples)
    else:
        if not os.path.exists(mun_path):
            sys.stderr.write('Missing file: %s\n' %mun_path)
        if not os.path.exists(semcor_path):
            sys.stderr.write('Missing file: %s\n' %semcor_path)
        return None

def variation_experiment():
    print('=' * 50)
    print('Variation experiment')
    print('=' * 50)

    print('Small (h%dp%d):' %(SmallConfig.hidden_size, SmallConfig.emb_dims))
    output_dir = os.path.join('output', '2017-11-23-02d85a9')
    result_dirs = []
    for root, children, _ in os.walk(output_dir):
        for child in children:
            if child.startswith('results-seed-'):
                result_dirs.append(os.path.join(root, child))
    print_variation_results(result_dirs)
    
    print('H256P64:')
    output_dir = os.path.join('output', '2017-11-24-e93fdb2')
    result_dirs = []
    for child in os.listdir(output_dir):
        if re.match('lstm-wsd-gigaword-h256p64-seed_.*.results', child):
            result_dirs.append(os.path.join(output_dir, child))
    print_variation_results(result_dirs)
    
def print_variation_results(result_dirs):
    perf_list = []
    for result_dir in result_dirs:
        p = read_performance(result_dir)
        if p is not None:
            perf_list.append(p)
    df = pd.DataFrame(perf_list, columns=ModelPerformance._fields)
    print('All results:')
    print(df)
    print('Mean:', df['semcor'].mean(), df['mun'].mean())
    print('Std:', df['semcor'].std(), df['mun'].std())

def report_wsd_performance_vs_data_size():
    print('=' * 50)
    print('Data size experiment')
    print('=' * 50)
    perf_list = []
    for ds_percent in (1, 10):
        base_path = 'output/2017-11-24-4e4a04a/lstm-wsd-gigaword_%02d-pc_large' %ds_percent
        perf_list.append(read_performance(base_path + '.results', name=ds_percent))
    df = pd.DataFrame(perf_list, columns=ModelPerformance._fields)
    print(df)
    print()
    print(df.to_latex(index=False))
    
    log_paths = ['slurm-3788824.out', #'slurm-3792014.out', 'slurm-3792013.out',
                 'slurm-3785679.out', 'slurm-3785673.out']
    rows = []
    for log_path in log_paths:
        assert os.path.exists(log_path), "Please adjust the log paths to your system"
        with open(log_path) as f:
            s = f.read()
        ds_percent = int(re.search(r'Saved best model to output/.+/lstm-wsd-gigaword_(\d+)-pc_large-best-model', s).group(1))
        dev_cost = min(float(val) for val in re.findall('Dev cost: ([\d\.]+)', s))
        train_cost = min(float(val) for val in re.findall('Train cost: ([\d\.]+)', s))
        rows.append((ds_percent, math.exp(train_cost), math.exp(dev_cost)))
    rows.sort()
    csv_writer = csv.writer(sys.stdout)
    csv_writer.writerow(['Data size', 'Train NLL', 'Valid NLL'])
    csv_writer.writerows(rows)

if __name__ == '__main__':
    report_wsd_performance_vs_data_size()
#     variation_experiment()
    