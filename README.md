# Repo for LDL for different cosmologies

This is the repo for my current project of applying LDL to different cosmologies.

LDL is [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926), and the code I use is based on [this](https://github.com/biweidai/LDL).

Disclaimer: At the moment the repo contains mostly images and code to generate them, as that's all I've been able to do up to this point.

## Packages

The packages needed for running LDL are:

[vmad](https://github.com/rainwoodman/vmad)  
[nbodykit](https://github.com/bccp/nbodykit)  
[fastpm-python](https://github.com/rainwoodman/fastpm-python)

## Important notes

Note that installing the packages above with the procedure indicated in the READMEs results in errors when running the code. Specifically, it is recommended to install nbodykit as indicated in the [guide](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html) rather than in the fastpm-python README, as in the latter case conda doesn't solve the environment for mpi4py properly and this causes errors. Additionally, to be compatible with the latter versions of numpy, ```pmesh/domain.py``` should have

```python
self.edges = [numpy.asarray(g) for g in edges]
```
at line 380, while ```nbodykit/source/mesh/linear.py``` should have

```python
mask = numpy.bitwise_and.reduce(np.array([ki == 0 for ki in k], dtype=object))
```

at line 87, see [this](https://github.com/rainwoodman/fastpm-python/issues/18).

The vmad code installed via ```pip install vmad``` is outdated and will yield errors. Clone the repo and update the package manually.

Originally I solved the mpi4py problem by installing mpich with apt, locating ```libmpich.so.12``` and creating a symbolic link from it to ```libmpi.so.12```; however, this may yield errors on some OS.

## Docker

The attached Dockerfile builds an image from ubuntu installing nbodykit, vmad and fastpm correctly on a conda environment called LDLenv. It also installs matplotlib and Pylians for visualization. Build the image with (may take several minutes, mostly due to installing nbodykit and Pylians)

```
docker build -t NAME[:TAG] /PATH/TO/DOCKERFILE/
```

and run it with (suggested)

```
docker run -it NAME[:TAG]
```

The image is also available at ```maurorigo/ldlimg:final```.

**NOTE**: on Mac with M1 or above chips, you need to specify ```--platform linux/x86_64``` both on when building and running.

Additional files in the Docker folder can be used to test the image created. They can be loaded during building with the ```COPY``` command. Otherwise, they can be loaded during run with ```docker cp``` and ```docker commit``` or simply by creating the corresponding files in the image and copy-pasting the text.

To test fastpm, you can load  ```testfastpm.py``` and ```input_spectrum_PLANCK15.txt``` and run:

```
python testfastpm.py 0
```

or, with MPI:

```
mpirun -n PROCS python testfastpm.py z
```

where the ```z``` is the redshift of the snapshot to create (which will be saved in the current folder) and ```PROCS``` is the number of MPI processes.

The image can also be used to build a singularity image that can run for instance on [Leonardo](https://leonardo-supercomputer.cineca.eu/). However, in that case it is necessary to run ```conda init bash``` and restarting the image before using conda. 

## Actions

At the moment, actions only test whether the image works correctly in running a fastpm simulation and plotting a DM overdensity field, as I did not have time to do other things. In the future, actions should also include a test run of the N-body simulator itself (different simulators with different codes may be used, but this has not been fully defined yet), and testing of the LDL model.
