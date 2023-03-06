FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ssh \
   && apt-get upgrade -y --no-install-recommends \
        libstdc++6 \
   && apt-get purge -y software-properties-common \
   && rm -rf /var/lib/apt/lists/* \
   && apt-get autoremove -y --purge \
   && apt-get clean

RUN apt-get update -y \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get autoremove -y --purge \
    && apt-get clean

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

#RUN pip install protobuf==3.20.*
#RUN pip install black[jupyter]
RUN pip install jupyter_contrib_nbextensions
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip
RUN jupyter nbextension enable jupyter-black-master/jupyter-black
RUN pip install "google-cloud-storage>=1.13.2"
#COPY clearml.conf /root
