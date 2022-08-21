FROM tensorflow/tensorflow:latest
LABEL maintainer="kodamanbou0424@gmail.com"
LABEL version="1.0"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git \
                       wget

WORKDIR /work
ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/kodamanbou/sEMG-Net.git
WORKDIR /work/sEMG-Net

CMD [ "/bin/sh"]
