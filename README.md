# ocean_modelling

## Tridiagonal solver
The different implementations of the tridiagonal solver can be found in the [tridiag](tridiag) folder. The C++/CUDA code in the [cpp](tridiag/cpp) subfolder.

## Superbee

The optimized superbee kernels are in [this](tridiag/jax_xla/cuda_superbee_kernels.cu) file.

## turbulent kinetic energy

The futhark code for this routine is in [this](turbulent_kinetic_energy/tke.fut) file.

## Interfacing CUDA with Jax through XLA
The code for interfacing CUDA with Jax is located in [this folder](tridiag/jax_xla) and can be installed with the command ```pip install -e jax_xla```
Of course this needs the Jax (https://github.com/google/jax) package to be installed.
For compiling the CUDA kernels through the python setuptools I used code from [this](https://github.com/rmcgibbo/npcuda-example) repository. 
And similar integration into Jax can also be found [here](https://github.com/PhilipVinc/mpi4jax/tree/master/mpi4jax/cython).

## Please ignore the rest of this repository.
