# running from tensorflow docker base image
FROM tensorflow/tensorflow:1.15.5

RUN useradd fuller -m
USER fuller

RUN cd /home/fuller/
WORKDIR /home/fuller/