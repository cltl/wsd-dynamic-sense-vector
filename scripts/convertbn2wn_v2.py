import utils
logger = utils.start_logger('log.txt')

logger.info('starting')


import multiprocessing
from glob import glob
from utils import get_instances
import shutil
import os


# https://stackoverflow.com/questions/9038711/python-pool-with-worker-processes
logger.info('loaded precomputed dictionaries')

class Worker(multiprocessing.Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue= queue

    def run(self):
        for path in iter(self.queue.get, None):
            worker_id = multiprocessing.current_process().name
            get_instances(path, worker_id, debug=False)


# rm and mkdir
for dir_ in ['output/hdn', 'output/synset']:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)


num_workers = 1
request_queue = multiprocessing.Queue()
processes = []
for _ in range(num_workers):
    p = Worker(request_queue)
    processes.append(p)
    p.start()

logger.info('started workers(s)')

# main_input_folder = '/mnt/scistor1/group/marten/babelfied-wikipediaXML/'
main_input_folder = 'input'
for path in glob(main_input_folder + '/14/*.xml.gz'):
    request_queue.put(path)


# kill workers
logger.info('killing workers')
for i in range(num_workers):
    request_queue.put(None)


logger.info('done')




