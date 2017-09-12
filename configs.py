import tensorflow as tf
import os

preprocessing_script_revision = '694cb4d'
preprocessed_data_dir = os.path.join('preprocessed-data', preprocessing_script_revision)
os.makedirs(preprocessed_data_dir, exist_ok=True)

gigaword_path = 'data/gigaword'
preprocessed_gigaword_path = os.path.join(preprocessed_data_dir, 'gigaword.txt')

class DefaultConfig(object):
    vocab_size = 10**6 + 3
    max_grad_norm = 5
    num_senses = 4
    float_dtype = tf.float32
    training_on_same_length_sentences = True
    sampled_softmax = True
    optimized_batches = True

class SmallConfig(DefaultConfig):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.1
    hidden_size = 100
    max_epoch = 100
    emb_dims = 10
    max_stagnant_count = 5

class MediumConfig(DefaultConfig):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.1
    hidden_size = 256
    max_epoch = 500
    emb_dims = 64
    max_stagnant_count = 10

class LargeConfig(DefaultConfig):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.1
    hidden_size = 512
    max_epoch = 1000
    emb_dims = 128
    max_stagnant_count = 20

class GoogleConfig(DefaultConfig):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.1
    hidden_size = 2048
    max_epoch = 2000
    emb_dims = 512
    max_stagnant_count = 50

class TestConfig(DefaultConfig):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 0.1
    hidden_size = 2
    max_epoch = 1
    batch_size = 20
    max_stagnant_count = -1

def get_config():
    FLAGS = tf.flags.FLAGS
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "google":
        return GoogleConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
