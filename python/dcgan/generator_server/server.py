import base64
from io import BytesIO
from flask import Flask
from .generator import Generator
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf

app = Flask(__name__)
gen = Generator()


@app.route("/")
def index():
    data = gen.generate()[0]
    data = tf.math.abs(data)
    img = array_to_img(data, data_format="channels_last")

    byte_buffer = BytesIO()
    img.save(byte_buffer, format="PNG")
    img_str = base64.b64encode(byte_buffer.getvalue())
    decoded = img_str.decode("utf-8")

    return f"<img src='data:image/png;base64, {decoded}'></img>"
