FROM ubuntu:latest

# Ubuntu image with nbodykit, vmad and fastpm installed. Works at least up to numpy 1.25.2.
# Also installs visualization packages (matplotlib and Pylians)

# Install wget
RUN apt-get update \
    && apt-get install -y wget bzip2 ca-certificates curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install vim
RUN apt-get update && apt-get install -y vim

# Install miniconda3
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh

# Use conda shell to create environment and install packages
RUN . /root/miniconda3/etc/profile.d/conda.sh \
    && conda create -n LDLenv python=3.8 \
    && conda activate LDLenv \
    && conda install -y cython numpy scipy mpi4py nose \
    && conda install -y -c bccp nbodykit \
    && pip install -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip \
    && pip install fastpm

# Install vmad (the version available with pip is outdated and yields errors)
RUN cd /home \
    && git clone https://github.com/rainwoodman/vmad.git \
    && cp -r vmad/vmad /root/miniconda3/envs/LDLenv/lib/python3.8/site-packages \
    && rm -r vmad

# Nasty trick to have libmpi.so.12 (mpi4py requires it and openmpi doesn't distribute it anymore)
RUN apt install -y mpich \
    && ln -s /usr/lib/x86_64-linux-gnu/libmpich.so.12 /usr/lib/x86_64-linux-gnu/libmpi.so.12

# Install visualization tools (after mpi otherwise Pylians complains)
RUN . /root/miniconda3/etc/profile.d/conda.sh \
    && conda activate LDLenv \
    && pip install matplotlib \
    && pip install Pylians

# Edit some lines of files in libraries for compatibility issues
RUN cd /root/miniconda3/envs/LDLenv/lib/python3.8/site-packages/nbodykit/source/mesh/ \
    && sed -i '87s/.*/            mask = numpy.bitwise_and.reduce(np.array([ki == 0 for ki in k], dtype=object))/' linear.py \
    && cd /root/miniconda3/envs/LDLenv/lib/python3.8/site-packages/pmesh \
    && sed -i '380s/.*/        self.edges = [numpy.asarray(g) for g in edges]/' domain.py

# Just print some info
RUN echo 'echo "This image runs on Ubuntu and has a conda environment called \"LDLenv\" that has installed"' > /usr/local/bin/start.sh
RUN echo 'echo "the packages necessary to run the LDL code at https://github.com/biweidai/LDL."' >> /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Initialize conda so that it's available next use
RUN ~/miniconda3/bin/conda init bash

WORKDIR /home

ENTRYPOINT /usr/local/bin/start.sh && /bin/bash 
