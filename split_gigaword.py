from sklearn.model_selection._split import train_test_split
from version import version
from subprocess import check_output
import gzip
from tqdm import tqdm

if __name__ == '__main__':
    inp_path = 'output/gigaword-deduplicated.2018-05-10-f802d4d.txt.gz'
    out_pattern = 'output/gigaword-%%s.%s.txt.gz' %version
    
    num_lines = int(check_output('zcat %s | wc -l' %inp_path, shell=True))
    sent_indices = list(range(num_lines))
    train_indices, dev_indices = train_test_split(sent_indices, test_size=0.1)
    train_indices = set(train_indices)
    dev_indices = set(dev_indices)
    
    with gzip.open(inp_path, 'rt', encoding='utf-8') as f_inp, \
            gzip.open(out_pattern %'train', 'wt', encoding='utf-8') as f_train, \
            gzip.open(out_pattern %'dev', 'wt', encoding='utf-8') as f_dev:
        inp_lines = tqdm(f_inp, total=num_lines, unit='line', 
                         desc='Splitting GigaWord')
        for i, line in enumerate(inp_lines):
            if i in train_indices:
                f_train.write(line)
            elif i in dev_indices:
                f_dev.write(line)
    print('Result written to %s' %(out_pattern %'{train,dev}'))
