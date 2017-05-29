import multiprocessing
from glob import glob
from utils import get_instances

class Worker(multiprocessing.Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue= queue

    def run(self):
        for path in iter(self.queue.get, None):
            worker_id = multiprocessing.current_process().name
            get_instances(path, worker_id)

num_workers = 8
request_queue = multiprocessing.Queue()
for _ in range(num_workers):
    Worker(request_queue).start()

main_input_folder = '/mnt/scistor1/group/marten/babelfied-wikipediaXML/'
for path in glob(main_input_folder + '/**/*.xml.gz'):
    request_queue.put(path)