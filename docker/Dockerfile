# running from tensorflow docker base image
FROM tensorflow/tensorflow:1.15.5

# user configurations
RUN useradd fuller -m
USER fuller

RUN cd /home/fuller/
WORKDIR /home/fuller/

# upgrade pip
RUN pip install --upgrade pip
RUN pip install mpes fuller
RUN pip install --upgrade https://github.com/VincentStimper/mclahe/archive/master.zip

# running entire fuller package in docker image
RUN mkdir scripts
COPY scripts scripts

# run in interactive mode
CMD ["python3", "mpes_reconstruction.py"]