# running from tensorflow docker base image
FROM tensorflow/tensorflow:1.15.5

RUN mkdir fuller_scripts
WORKDIR /fuller_scripts/

# upgrade pip
RUN pip install --upgrade pip
RUN pip install mpes fuller
RUN pip install --upgrade https://github.com/VincentStimper/mclahe/archive/master.zip

RUN mkdir data
COPY data data

# running entire fuller package in docker image
COPY scripts ./

# run in interactive mode
CMD ["python3", "mpes_reconstruction.py"]