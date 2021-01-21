# ocean_modelling

## Tridiagonal solver
The different implementations of the tridiagonal solver can be found in the ```tridiag``` folder. The C++/CUDA code in the ```tridiag/cpp``` subfolder.

## Superbee

The optimized superbee kernels are in the ```tridiag/jax_xla/superbee_kernels.cu``` file.

## turbulent kinetic energy

The futhark code for this routine is in the ```turbulent_kinetic_energy/tke.fut``` file.

# Interfacing CUDA with Jax through XLA
The code for interfacing CUDA with Jax is located in ```tridiag/jax_xla``` and can be installed with the command ```pip install -e jax_xla```
Of course this needs the Jax (https://github.com/google/jax) package to be installed.
