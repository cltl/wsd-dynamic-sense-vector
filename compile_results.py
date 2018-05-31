import os
from collections import namedtuple
import re
import csv
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # use this when run on server
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.linear_model.base import LinearRegression
import configs
from configs import SmallConfig, H256P64, LargeConfig, GoogleConfig,\
    DefaultConfig
from glob import glob

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

    from configs import SmallConfig
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
    

def read_json_files(paths):
    jsons = []
    for path in paths:
        with open(path) as f:
            jsons.extend(f.readlines())
    json = '[%s]' %','.join(jsons)
    df = pd.read_json(json, orient='records')
    df['path'] = paths
    return df


def draw_data_size_vs_performance_chart():
    ''' Create figure for paper '''
    paths = glob('output/data-sizes/*.results/*/results.json') + \
            glob('output/model-h2048p512-mfs-true.results/*/results.json')
    df = read_json_files(paths)
    
    def parse_path(val):
        if 'model-h2048p512' in val:
            return 100
        else:
            return float(re.search(r'(\d+)\.results', val).group(1))
        
    df['data-pct'] = df['path'].apply(parse_path)
    df['words'] = 1.8e9 * df['data-pct']/100
    df = df.append([{'words': 1e11, 'model': 'Yuan et al. (T: SemCor)', "competition": "SemEval13", 'F1': 0.670},
                    {'words': 1e11, 'model': 'Yuan et al. (T: OMSTI)', "competition": "SemEval13", 'F1': 0.673},
                    {'words': 1e11, 'model': 'Yuan et al. (T: SemCor)', "competition": "Senseval2", 'F1': 0.736},
                    {'words': 1e11, 'model': 'Yuan et al. (T: OMSTI)', "competition": "SemEval13", 'F1': 0.673},
                    {'words': 1e11, 'model': 'Yuan et al. (T: SemCor)', "competition": "Senseval2", 'F1': 0.736},
                    {'words': 1e11, 'model': 'Yuan et al. (T: OMSTI)', "competition": "Senseval2", 'F1': 0.724}])
    print(df)
    
    def get_xy(competition, model): 
        sub_df = df[df['model'].str.contains(model, regex=False)]
        sub_df = sub_df.query('competition == "%s"' %competition).sort_values('words')
        return sub_df['words'], sub_df['F1']
    
    with PdfPages('output/data_size_vs_performance.pdf') as pdf:
        se13_semcor_handle, = plt.plot(*get_xy('SemEval13', '(T: SemCor)'), '-o', label='SemEval13 (T: SemCor)')
        se13_mun_handle, = plt.plot(*get_xy('SemEval13', '(T: SemCor+OMSTI)'), '--o', label='SemEval13 (T: OMSTI)')
        se2_semcor_handle, = plt.plot(*get_xy('Senseval2', '(T: SemCor)'), ':o', label='Senseval2 (T: SemCor)')
        se2_mun_handle, = plt.plot(*get_xy('Senseval2', '(T: SemCor+OMSTI)'), '-.o', label='Senseval2 (T: OMSTI)')
        plt.legend(handles=[se13_semcor_handle, se13_mun_handle, se2_semcor_handle, se2_mun_handle], loc='lower right')
        plt.axis([1.5e7, 1.1e11, 0, 1])
        plt.ylabel('F1')
        plt.xlabel('Tokens')
        plt.xscale('log')
        pdf.savefig()
        plt.show()
        plt.close()
    # extrapolate from data
    lr = LinearRegression()
    words, f1s = get_xy('SemEval13', 'Our LSTM (T: SemCor)')
    lr.fit(f1s.values.reshape([-1,1]), np.log10(words.values.reshape([-1,1])))
    print('Extrapolated data size (words):')
    print(lr.predict([[0.75], [0.8]]))


def compute_num_params(vocab_size, p, h):
    return (vocab_size*p*2 + # input and output embeddings
            p*h + h*h + h + # input gates
            p*h + h*h + h + # candidate states
            p*h + h*h + h + # forget gates
            p*h + h*h + h*h + h + # output gates
            p*h # context layer
            )    
    
    
def draw_capacity_vs_performance_chart():
    ''' Create figure for paper '''
    paths = glob('output/model-h*-mfs*/*/results.json')
    paths = [p for p in paths if 'mfs-true' in p]
    df = read_json_files(paths)
    def parse_path(val):
        m = re.search(r'model-h(\d+)p(\d+)', val)
        return pd.Series({'h': int(m.group(1)), 'p': int(m.group(2))})
    params = df['path'].apply(parse_path)
    vocab_size = configs.DefaultConfig.vocab_size
    df['num_params'] = compute_num_params(vocab_size, params['p'], params['h'])
    print(df)
    
    def get_xy(competition, model): 
        sub_df = df.query('competition == "%s" and model== "%s"'
                          %(competition, model)).sort_values('num_params')
        return sub_df['num_params'], sub_df['F1']
    
    with PdfPages('output/capacity_vs_performance.pdf') as pdf:
        se13_semcor_handle, = plt.plot(*get_xy('SemEval13', 'Our LSTM (T: SemCor)'), '-o', label='SemEval13 (T: SemCor)')
        se13_mun_handle, = plt.plot(*get_xy('SemEval13', 'Our LSTM (T: SemCor+OMSTI)'), '--o', label='SemEval13 (T: OMSTI)')
        se2_semcor_handle, = plt.plot(*get_xy('Senseval2', 'Our LSTM (T: SemCor)'), ':o', label='Senseval2 (T: SemCor)')
        se2_mun_handle, = plt.plot(*get_xy('Senseval2', 'Our LSTM (T: SemCor+OMSTI)'), '-.o', label='Senseval2 (T: OMSTI)')
        plt.legend(handles=[se13_semcor_handle, se13_mun_handle, se2_semcor_handle, se2_mun_handle], loc='lower right')
        plt.axis([1.9e7, 1.1e9, 0, 1])
        plt.ylabel('F1')
        plt.xlabel('Parameters')
        plt.xscale('log')
        pdf.savefig()
        plt.show()
        plt.close()
    # extrapolate from data
#     lr = LinearRegression()
#     lr.fit(df['semcor'].values.reshape([-1,1]), 
#            np.log10(df['data_size']).values.reshape([-1,1]))
#     print('Extrapolated data size:')
#     print(lr.predict([[0.75], [0.8]]))


def report_model_params():
    v = DefaultConfig.vocab_size
    models = [SmallConfig, H256P64, LargeConfig, GoogleConfig]
    table = [['%.0fM' %(v/10**6), m.emb_dims, m.hidden_size, 
              "%.0fM" %(compute_num_params(v, m.emb_dims, m.hidden_size)/10**6)]
              for m in models]
    df = pd.DataFrame(table, columns=['Vocab.', 'p', 'h', '#params'])
    print(df.to_latex(index=False))


def report_performance_google_model():
    paths = glob('output/model-h2048p512-mfs*/*/results.json')
    df = read_json_files(paths)
    for competition in ['Senseval2', 'SemEval13']:
        print('*** %s ***' %competition)
        print(df.loc[df['competition'] == competition][['model', '+MFS', 'P', 'R', 'F1']].sort_values(['model', '+MFS']))


if __name__ == '__main__':
#     report_wsd_performance_vs_data_size()
#     variation_experiment()
    draw_data_size_vs_performance_chart()
    draw_capacity_vs_performance_chart()
#     report_model_params()
#     report_performance_google_model()