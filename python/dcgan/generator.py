import os
import tensorflow as tf
import matplotlib.pyplot as plt
from DCGAN import DCGAN

generator = DCGAN.create_generator_model(100)
seed = tf.random.uniform([25, 100])

checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "dcgan")
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=5)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

predictions = generator(seed, training=False)
r, c = 5, 5
fig, axs = plt.subplots(r, c)
cnt = 0

for i in range(r):
    for j in range(c):
        axs[i, j].imshow(tf.math.abs(predictions[cnt, :, :, :]))
        axs[i, j].axis("off")
        cnt += 1

plt.savefig(os.path.join(os.getcwd(), 'cats.png'))
