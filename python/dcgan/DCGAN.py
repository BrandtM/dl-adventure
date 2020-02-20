import tensorflow as tf
from tensorflow.keras import layers


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
