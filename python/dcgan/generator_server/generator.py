import os
import tensorflow as tf
from ..DCGAN import DCGAN


class Generator:
    def __init__(self):
        self.generator = DCGAN.create_generator_model(100)

        self.checkpoint_dir = os.path.join(os.getcwd(), "server", "checkpoints")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "dcgan")

        self.checkpoint = tf.train.Checkpoint(generator=self.generator)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_prefix,
            max_to_keep=5
        )
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def generate(self):
        seed = tf.random.uniform([1, 100])
        return self.generator(seed, training=False)
