import tensorflow as tf

class DefaultConfig(object):
    vocab_size = 10**6 + 3
    max_grad_norm = 5
    num_senses = 4
    float_dtype = tf.float32

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
    hidden_size = 200
    max_epoch = 500
    emb_dims = 100
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
