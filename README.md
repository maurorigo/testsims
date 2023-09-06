# Repo for LDL for different cosmologies

This is the repo for my current project of applying LDL to different cosmologies.

LDL is [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926), and the code I use is based on [this](https://github.com/biweidai/LDL). 

## Packages

The packages needed for running LDL are:

[vmad](https://github.com/rainwoodman/vmad)  
[nbodykit](https://github.com/bccp/nbodykit)  
[fastpm-python](https://github.com/rainwoodman/fastpm-python) 

## Important notes

Note that installing the packages above with the procedure indicated in the READMEs results in errors when running the code mostly due to updates of numpy. Specifically, ```pmesh/domain.py``` should have

```python
self.edges = [numpy.asarray(g) for g in edges]
```
at line 380, while ```nbodykit/source/mesh/linear.py``` should have

```python
mask = numpy.bitwise_and.reduce([ki == 0 for ki in k])
```

at line 87, see [this](https://github.com/rainwoodman/fastpm-python/issues/18).

## Actions

At the moment, actions only test wether LDL.py works as intended when producing a stellar map. In the future, it should also include a test run of the N-body simulator (different simulators with different codes may be used, but this has not been fully defined yet).
