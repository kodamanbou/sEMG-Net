FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="kodamanbou0424@gmail.com"
LABEL version="1.0"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git \
                       wget

WORKDIR /work
ADD requirements.txt .
RUN pip3 install -r requirements.txt

CMD [ "/bin/sh"]
