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

# modify source code
COPY ../scripts/mrfRec.py /home/fuller/.local/lib/python3.6/site-packages/fuller/mrfRec.py

# run in interactive mode
CMD ["python3", "mpes_reconstruction.py"]