import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from python.dcgan import DCGAN


class GanTrainer:
    def __init__(self, image_size, epochs, batch_size, batch_count):
        self.image_size = image_size
        self.epochs = epochs
        self.noise_dim = 100
        self.seed = tf.random.uniform([25, 100])
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.data_path = os.path.join(os.getcwd(), "train_data", "gan")
        self.image_count = 48

        self.generator = DCGAN.create_generator_model(self.image_size)
        self.discriminator = DCGAN.create_discriminator_model(self.image_size)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "dcgan_100")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "dcgan_100-356")

        datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.0,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Remember that images need to reside in another subdirectory under train_data/gan
        self.image_generator = datagen.flow_from_directory(
            os.path.join(os.getcwd(), "train_data", "gan"),
            batch_size=self.batch_size,
            class_mode=None,
            target_size=(self.image_size, self.image_size)
        )

        self.checkpoint.restore(checkpoint_path)

    @tf.function
    def train_step(self):
        noise = tf.random.uniform([self.batch_size, 100])

        for _ in range(self.batch_count):
            image_batch = next(self.image_generator)
            image_batch /= 255

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(image_batch, training=True)
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
            self.train_step()
            end = time.time() - start

            if epoch % 100 == 0:
                print(f"Time for epoch {epoch} was {end} seconds")

            if epoch % 1000 == 0:
                save_path = self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Saved checkpoint for epoch {epoch} to {save_path}")

        save_path = self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print(f"Saved checkpoint to {save_path}")
        self.generate_image()

    def generate_image(self, ):
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
