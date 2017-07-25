import subprocess

def count_lines_fast(path, block_size=65536):
    '''
    Credit: glglgl (https://stackoverflow.com/a/9631635/217802)
    Might miss out the last line but it doesn't matter for a huge file such as Gigaword
    '''
    total_lines = 0
    with open(path) as f:
        while True:
            bl = f.read(block_size)
            if not bl: break
            total_lines += bl.count("\n")
    return total_lines

def generate_data_size_experiments():
    name_template = 'train-lstm-wsd-{percent}pc-data-google-model.job'
    content_template = '''#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda80/toolkit
module load cuda80/blas
module load cuda80
module load cuDNN

echo -n 'Started: ' && date

head -n {num_lines} output/gigaword.txt > output/gigaword_{percent}pc.txt && \\
        python3 -u prepare-lstm-wsd.py output/gigaword_{percent}pc.txt \\
                                       output/gigaword_{percent}pc-lstm-wsd && \\
        python3 -u train-lstm-wsd.py --model google \\
                                     --data_path output/gigaword_{percent}pc-lstm-wsd \\
                                     --save_path output/lstm-wsd-google_trained_on_gigaword_{percent}pc

echo -n 'Finished: ' && date
'''
    total_lines = count_lines_fast('output/gigaword.txt')
    for percent in (1, 10, 50, 75):
        num_lines = int(percent / 100.0 * total_lines)
        fname = name_template.format(**locals())
        with open(fname, 'w') as f_script:
            f_script.write(content_template.format(**locals()))
        subprocess.call('chmod a+x %s' %fname, shell=True)

if __name__ == '__main__':
    generate_data_size_experiments()