# Wind farm optimization by genetic algorithms
([versão em português](ptREADME.md))

## Description

Two wind turbines may produce less energy when close to each other than when they are far apart. This is due to the fact that the wind slows down after interacting with one turbine and, therefore, the other turbine might generate less energy depending of its position. Because of this, a small wind farm with many turbines is inefficient. However, a large wind farm with few turbines far apart is also inefficient. Given the direction and speed distributions of the wind in a region and a limited ground area, it is possible to optimize the placement of the wind turbines to maximize the energy output and minimize costs.

This repository contains python code for simulation and optimization of wind farms. Single-objective (as in Emami and Noghreh[^1]) and multi-objective optimizations by genetic algorithms are implemented.

## Tools and Methods

[![Skills](https://skillicons.dev/icons?i=python)](google.com)

[Numpy](https://numpy.org/)

[Matplotlib](https://matplotlib.org/)

Genetic algorithms

NSGA-II (multi-objective selection)

## Results


## Takeaways

## Files

- [windfarm.py](windfarm.py): definition of the class Windfarm for description, simulation, and visualization of wind farms.
- [windfarm_examples.ipynb](windfarm_examples.ipynb): examples of usage of Windfarm class.
- [matrix_operators.py](matrix_operators.py): definition of matrix genetic operators.
- [operators_examples.ipynb](operators_examples.ipynb): examples of usage of operators.

[^1]: A. Emami and P. Noghreh, “New approach on optimization in placement of wind turbines within wind farm by genetic algorithms,” Renew. Energy, vol. 35, no. 7, pp. 1559–1564, 2010, doi: 10.1016/j.renene.2009.11.026.
