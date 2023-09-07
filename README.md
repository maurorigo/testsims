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

Note that ***installing the packages above with the procedure indicated in the READMEs results in errors when running the code***. Specifically, ```pmesh/domain.py``` should have

```python
self.edges = [numpy.asarray(g) for g in edges]
```
at line 380, while ```nbodykit/source/mesh/linear.py``` should have

```python
mask = numpy.bitwise_and.reduce(np.array([ki == 0 for ki in k], dtype=object))
```

at line 87, see [this](https://github.com/rainwoodman/fastpm-python/issues/18).

The vmad code installed via ```pip install vmad``` is outdated and will yield errors. Clone the repo and update the package manually.

Additionally, there may be problems with mpi4py, as the installed version may require a file named ```libmpi.so.12``` which apparently used to come with OpenMPI v1.10, which is no longer supported. A workaround consists in installing mpich, locating ```libmpich.so.12``` and creating a symbolic link from it to ```libmpi.so.12```.

## Docker

The attached Dockerfile builds an image from ubuntu installing nbodykit, vmad and fastpm correctly on a conda environment called LDLenv. Additional files in the Docker folder can be used to test the image created.

## Actions

At the moment, actions only test wether LDL.py and model.py work as intended when producing a stellar map. In the future, it should also include a test run of the N-body simulator (different simulators with different codes may be used, but this has not been fully defined yet).
