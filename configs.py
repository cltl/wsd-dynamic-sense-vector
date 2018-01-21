import os
from version import version

output_dir = os.path.join('output', version)
os.makedirs(output_dir, exist_ok=True)

gigaword_path = 'data/gigaword'

special_symbols = ['<target>', '<unkn>', '<pad>', '<eos>']

class DefaultConfig(object):
    vocab_size = 10**6 + len(special_symbols)
    max_grad_norm = 5
    num_senses = 4
    init_scale = 0.1
    learning_rate = 0.1
    assume_same_lengths = True
    sampled_softmax = True
    optimized_batches = True
    max_stagnant_count = 10
    max_epoch = 100
#     max_epoch = 1 # for debugging

class SmallConfig(DefaultConfig):
    hidden_size = 100
    emb_dims = 10

class H256P64(DefaultConfig):
    hidden_size = 256
    emb_dims = 64

class LargeConfig(DefaultConfig):
    hidden_size = 512
    emb_dims = 128

class GoogleConfig(DefaultConfig):
    hidden_size = 2048
    emb_dims = 512

class TestConfig(DefaultConfig):
    """Tiny config, for testing."""
    hidden_size = 2
    max_epoch = 1
    batch_size = 20

def get_config(FLAGS):
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "h256p64":
        return H256P64()
    elif FLAGS.model == "large" or FLAGS.model == "h512p128":
        return LargeConfig()
    elif FLAGS.model == "google" or FLAGS.model == "h2048p512":
        return GoogleConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
