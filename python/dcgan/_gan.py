"""
This was my initial GAN test using Keras only. It doesn"t work and never did. I do not definitively know why,
but these models produce random noise. I think it"s the way that I built the adversarial model.
This way of training is probably why using pure tensorflow with manual gradient application works flawlessly.
This file is just an archive for myself.
"""


import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten, Reshape, Conv2DTranspose, \
    BatchNormalization, UpSampling2D, LeakyReLU
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

discriminator_model_path = os.path.join(os.getcwd(), "models", "gan", "gan01_discriminator.h5")
generator_model_path = os.path.join(os.getcwd(), "models", "gan", "gan01_generator.h5")


class GAN:
    def __init__(self):
        self.batch_size = 2
        self.latent_dim = 100

        self.discriminator = self.build_discriminator()
        self.discriminator_model = self.build_discriminator_model()
        self.generator = self.build_generator()
        self.adversarial = self.build_adversarial_model()
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        self.image_generator = self.datagen.flow_from_directory(
            os.path.join(os.getcwd(), "train_data", "gan2"),
            batch_size=32,
            class_mode="binary",
            target_size=(200, 200)
        )


    def create_train_data(self):
        x_train = []
        y_train = []

        path = os.path.join(os.getcwd(), "train_data", "gan2")

        for i in range(20):
            x = load_img(os.path.join(path, "real", f"{i}.jpg"))
            x = img_to_array(x)
            # x /= 255
            x = cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            x = np.reshape(x, (1, 32, 32, 3))
            x_train.append(x)
            y_train.append(1)

        # for i in range(1, 8):
        #     x = load_img(os.path.join(path, "fake", f"{i}.jpg"))
        #     x = img_to_array(x)
        #     x = cv2.resize(x, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        #     x = np.reshape(x, (1, 128, 128, 3))
        #     x_train.append(x)
        #     y_train.append(0)

        xx = np.array(x_train)
        xx = np.reshape(xx, (20, 32, 32, 3))
        yy = np.array(y_train)
        yy = np.reshape(yy, (20, 1))
        return xx, yy


    def train(self, epochs):
        # self.discriminator_model.fit_generator(
        #     self.image_generator,
        #     steps_per_epoch=8,
        #     epochs=1)

        d_loss = 0
        a_loss = 0
        x_train, y_train = self.create_train_data()
        for epoch in range(epochs):
            input_noise = np.random.uniform(0, 1, (self.batch_size, self.latent_dim))
            fake_img = self.generator.predict(input_noise)
            d_loss_2 = self.discriminator_model.train_on_batch(x_train, y_train)
            g_loss = self.generator.train_on_batch(input_noise, np.reshape(np.ones(self.batch_size * 32 * 32 * 3),
                                                                           (self.batch_size, 32, 32, 3)))
            d_loss = self.discriminator_model.train_on_batch(fake_img, np.zeros(self.batch_size))
            # d_loss = 0.5 * np.add(d_loss_2, d_loss)

            # if epoch % 10 == 0:
            # self.discriminator_model.fit_generator(
            #     self.image_generator,
            #     steps_per_epoch=4,
            #     epochs=1)

            input_noise = np.random.uniform(0, 1, (self.batch_size, self.latent_dim))
            a_loss = self.adversarial.train_on_batch(input_noise, np.ones(self.batch_size))

            # if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(
                f"Discriminator (loss/accuracy) (real): {d_loss_2}\nDiscriminator (loss/accuracy) (fake): {d_loss}\nAdversarial loss (loss/accuracy): {a_loss}\n\nGenerator loss (loss/accuracy): {g_loss}\n\n")

    # print(f"Discriminator (loss/accuracy) : {d_loss}\nAdversarial loss (loss/accuracy): {a_loss}")

    def plot(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        imgs = self.generator.predict(noise)
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                red = imgs[cnt, :, :, 0]
                green = imgs[cnt, :, :, 1]
                blue = imgs[cnt, :, :, 2]
                axs[i,j].imshow(red+green+blue, cmap="hsv")
                axs[i,j].axis("off")
                cnt += 1

        plt.show()

        # input_noise = np.random.uniform(0, 1, (1, 100))
        # fake_img = self.generator.predict(input_noise)[0]
        # fake_img = 0.5 * fake_img + 0.5
        # plt.imshow(fake_img)
        # plt.show()

    def build_adversarial_model(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        # optimizer = RMSprop(lr=0.0001, decay=3e-8)
        optimizer = Adam(0.0002, 0.1)
        model.compile(optimizer=optimizer,
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(128, (3, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(256, (3, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        return model

    def build_discriminator_model(self):
        model = Sequential()
        model.add(self.discriminator)
        # optimizer = RMSprop(lr=0.0006, decay=9e-8)
        optimizer = Adam(0.0002, 0.5)
        model.compile(optimizer=optimizer,
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def build_generator(self):
        model = Sequential()

        model.add(Dense(32*32*3, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        model.add(Reshape((32, 32, 3)))

        model.add(Conv2D(512, 3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        model.add(Conv2D(256, 3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        model.add(Conv2D(128, 3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        model.add(Conv2D(3, 3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("tanh"))
        optimizer = Adam(0.0002, 0.5)
        model.compile(optimizer=optimizer,
                      loss="binary_crossentropy",
                      metrics=["accuracy"])



        # model.add(Dense(32, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(64))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(512))
        # # model.add(Activation("relu"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(1024))
        # # model.add(Activation("relu"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(2048))
        # # model.add(Activation("relu"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(128 * 128 * 3))
        # model.add(Activation("tanh"))
        # model.add(Reshape((128, 128, 3)))

        return model


if __name__ == "__main__":
    gan = GAN()
    gan.train(200)
    gan.plot()
