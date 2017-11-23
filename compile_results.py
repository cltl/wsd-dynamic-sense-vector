import os
import numpy as np

def variation_experiment():
    output_dir = os.path.join('output', '2017-11-23-02d85a9')
    semcor_results = []
    mun_results = []
    for root, _, fnames in os.walk(output_dir):
        for fname in fnames:
            path = os.path.join(root, fname)
            if root.endswith('semcor') and fname == 'results.txt':
                with open(path) as f:
                    semcor_results.append(int(f.read()))
            elif root.endswith('mun') and fname == 'results.txt':
                with open(path) as f:
                    mun_results.append(int(f.read()))
    print('=' * 50)
    print('Variation experiment')
    print('=' * 50)
    semcor_results = np.array(semcor_results, dtype=np.float32) / 1644
    mun_results = np.array(mun_results, dtype=np.float32) / 1644
    print('SemCor results: %s' %list(semcor_results))
    print('\tMean = %.3f, Std = %.3f' %(semcor_results.mean(), semcor_results.std()))
    print('MUN (OMSTI) results: %s' %list(mun_results))
    print('\tMean = %.3f, Std = %.3f' %(mun_results.mean(), mun_results.std()))
    

if __name__ == '__main__':
    variation_experiment()