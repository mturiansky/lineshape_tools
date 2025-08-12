# lineshape_tools
<!-- index start -->

`lineshape_tools` is a Python software that implements the formulation for evaluating the effects of electron-phonon coupling on the optical properties of defects.
In particular, it implements the approach pioneered by [Alkauskas *et al.*](https://doi.org/10.1088/1367-2630/16/7/073026) to compute the lineshape function within Huang-Rhys theory.
The code interfaces with [`mace`](https://mace-docs.readthedocs.io/en/latest/) and [`phonopy`](https://phonopy.github.io/phonopy/) to evaluate the dynamical matrix and obtain the phonons of a defect-containing supercell.

### Installation
<!-- install start -->
To install the latest version of `lineshape_tools`, create a new virtual environment and run
```
pip install lineshape_tools
```
<!-- install end -->
For more installation information and some performance considerations, see the [Installation page]().

### Usage 
`lineshape_tools` provides a command-line interface for interacting with the code. See
```
lineshape_tools --help
```
Detailed usage information can be found in the [Tutorials page]().

### How to Cite
<!-- cite start -->
If you use this code, please consider citing
```bibtex
@misc{turiansky_machine_2025,
  title = {Machine Learning for Fast and Accurate Optical Lineshapes of Defects},
  author = {Turiansky, Mark E. and Lyons, John L., and Bernstein, Noam},
  year = {2025},
  number = {arXiv:XXXX.XXXXX},
  eprint = {XXXX.XXXXX},
  primaryclass = {cond-mat},
  publisher = {arXiv},
  doi = {},
  urldate = {},
  archiveprefix = {arXiv},
}
```
<!-- cite end -->
Please also consider citing the foundational works that made this code possible on the [Citation page]().

<!-- index end -->
