FROM tensorflow/tensorflow:latest-py3

RUN pip install matplotlib pillow flask gunicorn

WORKDIR /application

# CMD gunicorn -b 0.0.0.0:5001 generator_server.wsgi:main
ENV FLASK_APP=/application/generator_server/server.py
CMD flask run --host=0.0.0.0