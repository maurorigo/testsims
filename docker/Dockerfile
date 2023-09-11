FROM ubuntu:latest

# Ubuntu image with nbodykit, vmad and fastpm installed. Works at least up to numpy 1.25.2.
# Also installs visualization packages (matplotlib and Pylians)

# Also works for singularity
# Uses MPICH, hopefully it's not a problem

# Install wget
RUN apt-get update \
    && apt-get install -y wget bzip2 ca-certificates curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install vim
RUN apt-get update && apt-get install -y vim

# Install gcc (necessary for Pylians)
RUN apt-get update \
    && apt install -y build-essential

# Install miniconda3 in /opt/
RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm -rf /opt/miniconda3/miniconda.sh

# Add conda to path to make things easy and create environment
ENV PATH=/opt/miniconda3/bin:$PATH
RUN conda create -n LDLenv -y python=3.8

# Create environment install packages
RUN . /opt/miniconda3/etc/profile.d/conda.sh \
    && conda activate LDLenv \
    && conda install -y -c bccp nbodykit \
    && conda update -c bccp --all \
    && pip install fastpm \
    && pip install Pylians \
    && pip install matplotlib

# Install vmad from git (the version available with pip is outdated and yields errors)
RUN cd /home \
    && git clone https://github.com/rainwoodman/vmad.git \
    && cp -r vmad/vmad /opt/miniconda3/envs/LDLenv/lib/python3.8/site-packages \
    && rm -r vmad

# Edit some lines of files in libraries for compatibility issues (works with installed version but not later)
RUN cd /opt/miniconda3/envs/LDLenv/lib/python3.8/site-packages/nbodykit/source/mesh/ \
    && sed -i '87s/.*/            mask = numpy.bitwise_and.reduce(np.array([ki == 0 for ki in k], dtype=object))/' linear.py \
    && cd /opt/miniconda3/envs/LDLenv/lib/python3.8/site-packages/pmesh \
    && sed -i '380s/.*/        self.edges = [numpy.asarray(g) for g in edges]/' domain.py

# Just print some info
RUN echo 'echo "This image runs on Ubuntu and has a conda environment called \"LDLenv\" that has installed"' > /usr/local/bin/start.sh \
    && echo 'echo "the packages necessary to run the LDL code at https://github.com/biweidai/LDL."' >> /usr/local/bin/start.sh \
    && chmod +x /usr/local/bin/start.sh

# Make sure any user will be able to access to conda
RUN chmod -R o+rX /opt/miniconda3

# Initialize conda
RUN conda init bash

ENTRYPOINT /usr/local/bin/start.sh && /bin/bash 

WORKDIR /home
