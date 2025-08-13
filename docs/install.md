# Installation

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- install start -->
:end-before: <!-- install end -->
```

Several libraries that are not strictly required but may be useful can be installed by appending `[extra]`:
```sh
pip install "lineshape_tools[extra]"
```

## Install from Github
`lineshape_tools` can be directly installed from github, providing access to potentially unreleased developments, with
```sh
pip install git+https://github.com/mturiansky/lineshape_tools.git
```

## Performance Considerations
All functionality directly related to `lineshape_tools` is implemented on the CPU.
While compatible with CPUs, training and evaluation of the `mace` models is significantly faster on a GPU.
We highly recommend using a GPU for those use cases.

### Numba
[`numba`](https://numba.readthedocs.io/en/stable/) is used under the hood to speed up several calculations related to lineshape evaluation.
There are a few things that can be done to speed up `numba` calculations in some instances.

##### SVML
On Intel processors, the short vector math library (SVML) can be enabled to speed up certain operations.
The runtime libraries from Intel are required for this.
On a `conda` installation, they should already be installed in the package `icc_rt`.
The `icc_rt` package is also available through pip
```sh
pip install icc_rt
```
However, you will likely need to add your virtual environment to the library path:
```sh
export LD_LIBRARY_PATH=/path/to/.virtualenvs/env_name/lib/:$LD_LIBRARY_PATH
```
This can be added to your `activate` script in your virtual environment (``/path/to/.virtualenvs/env_name/bin/activate``) to make the change persistent.
To check if the installation worked, run ``numba -s``; the output should include
```sh
   ...

   __SVML Information__
    SVML State, config.USING_SVML                 : True
    SVML Library Loaded                           : True
    llvmlite Using SVML Patched LLVM              : True
    SVML Operational                              : True

   ...
```

##### Environment Variables
There are also several environment variables for `numba` that can be enabled and may improve performance.
If your machine has AVX instructions, we recommend enabling it with:
```sh
export NUMBA_ENABLE_AVX=1
```
The full list of `numba` environment variables is available [here](https://numba.readthedocs.io/en/stable/reference/envvars.html).

### CuEquivariance
If utilizing the `mace` to train and evaluate the dynamical matrix on an NVIDIA GPU, it is recommend to use the cuEquivariance library to improve performance.
To install, run
```sh
pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
```
potentially replacing `cu12` with `cu11` if CUDA 11 is installed on the GPU.
For more information, please see [the `mace` documentation](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).
