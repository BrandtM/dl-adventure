import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow.keras import layers
import time

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def max_whole_divisions(n):
    """
    :type n: int
    :return: A tuple of (the lowest number achieved through divisions, the amount of divisions to get there)
    :rtype: (int, int)
    """
    divs = 0
    result = n / 2

    while result.is_integer():
        divs += 1
        result /= 2

    return int(result * 2), divs


class DCGAN:
    @staticmethod
    def get_cross_entropy():
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def create_generator_model(image_size):
        """
        Do not use a number that"s a power of 2.
        The models will be too big and saving checkpoints will take much longer.
        A value of 100 saves much faster and takes ~240MB of data per checkpoint.
        A value of 64 saves a lot slower and takes ~3.5GB of data per checkpoint.
        """
        start, divisions = max_whole_divisions(image_size)
        filters = 64 << divisions

        # we don"t want to start with a 1x1 convolution so we"ll make it a 2x2 one
        if start == 1:
            start = 2
            divisions -= 1

        model = tf.keras.Sequential()
        model.add(layers.Dense(start * start * filters, use_bias=False, input_dim=100))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((start, start, filters)))

        filters >>= 1
        model.add(layers.Conv2DTranspose(filters, (5, 5), (1, 1), "same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        for i in range(divisions - 1):
            filters >>= 1
            model.add(layers.Conv2DTranspose(filters, (5, 5), (2, 2), "same", use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), (2, 2), "same", use_bias=False, activation="tanh"))
        return model

    @staticmethod
    def create_discriminator_model(image_size):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), (2, 2), "same", input_shape=[image_size, image_size, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), (2, 2), "same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = DCGAN.get_cross_entropy()(tf.ones_like(real_output), real_output)
        fake_loss = DCGAN.get_cross_entropy()(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @staticmethod
    def generator_loss(fake_output):
        return DCGAN.get_cross_entropy()(tf.ones_like(fake_output), fake_output)


class GanTrainer:
    def __init__(self):
        self.image_size = 100
        self.epochs = 2000
        self.noise_dim = 100
        self.seed = tf.random.uniform([25, 100])
        self.batch_size = 48
        self.data_path = os.path.join(os.getcwd(), "train_data", "gan")
        self.image_count = 48

        self.generator = DCGAN.create_generator_model(self.image_size)
        self.discriminator = DCGAN.create_discriminator_model(self.image_size)
        self.data_set = self.create_train_data(self.image_count)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "dcgan_100")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "dcgan_100-286")
        self.checkpoint.restore(checkpoint_path)

    def create_train_data(self, image_count):
        x_train = []

        for i in range(image_count):
            # Remove this if you don't need to ignore images. If you have images named 000..001..002... you can use the
            # corresponding IDs here
            if i in (9, 45):
                continue

            x = load_img(os.path.join(self.data_path, "real", f"{i}.jpg"))
            x = img_to_array(x)
            x = cv2.resize(x, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            x = tf.reshape(x, (1, self.image_size, self.image_size, 3))
            x /= 255
            x_train.append(x)

        return tf.reshape(x_train, (image_count-2, self.image_size, self.image_size, 3))

    @tf.function
    def train_step(self, images):
        noise = tf.random.uniform([self.batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = DCGAN.generator_loss(fake_output)
            disc_loss = DCGAN.discriminator_loss(real_output, fake_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            self.train_step(self.data_set)
            end = time.time() - start

            if epoch % 100 == 0:
                print(f"Time for epoch {epoch} was {end} seconds")

            if epoch % 1000 == 0:
                save_path = self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Saved checkpoint for epoch {epoch} to {save_path}")

        save_path = self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print(f"Saved checkpoint to {save_path}")
        self.generate_image()

    def generate_image(self,):
        predictions = self.generator(self.seed, training=False)
        r, c = 5, 5
        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(tf.math.abs(predictions[cnt, :, :, :]))
                axs[i, j].axis("off")
                cnt += 1
        plt.show()


if __name__ == "__main__":
    trainer = GanTrainer()
    trainer.train()
