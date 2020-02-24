FROM tensorflow/tensorflow:latest-py3

RUN pip install matplotlib pillow flask gunicorn

WORKDIR /application

ENV PYTHONUNBUFFERED=TRUE
CMD gunicorn -b 0.0.0.0:5000 server.wsgi:app
