from glob import glob
import os
from multiprocessing.dummy import Pool as ThreadPool

import utils


logger = utils.start_logger('log.txt')

main_input_folder = '/mnt/scistor1/group/marten/babelfied-wikipediaXML/'
main_output_folder = 'output'

for input_folder in glob(main_input_folder + '*'):
    if os.path.isdir(input_folder):
        dir_name = os.path.basename(input_folder)

        if dir_name != '39':
            continue

        synset_output_path = main_output_folder + '/synset/' + dir_name + '.txt'
        hdn_output_path = main_output_folder + '/hdn/' + dir_name + '.txt'

        iterable = glob(input_folder + '/*.xml.gz')

        logger.info('starting with folder %s' % input_folder)

        pool = ThreadPool(20)
        all_generators = pool.map(utils.get_instances, iterable)

        num_docs = len(all_generators)
        synset_count = 0
        hdn_count = 0

        synset_file = open(synset_output_path, 'w')
        hdn_file = open(hdn_output_path, 'w')

        for generator in all_generators:
            for meaning_type, instance in generator:

                if meaning_type == 'synset':
                    synset_file.write(instance + '\n')
                    synset_count += 1
                elif meaning_type == 'hdn':
                    hdn_file.write(instance + '\n')
                    hdn_count += 1

        synset_file.close()
        hdn_file.close()

        logger.info('finished %s docs' % num_docs)
        logger.info('synset instances %s' % synset_count)
        logger.info('hdn instances %s' % hdn_count)
