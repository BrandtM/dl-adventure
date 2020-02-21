from python.dcgan import GanTrainer
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    trainer = GanTrainer(100, 100, 48)
    trainer.train()
