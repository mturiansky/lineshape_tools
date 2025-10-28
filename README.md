# lineshape_tools
<!-- index start -->

`lineshape_tools` is a Python software that implements the formulation for evaluating the effects of electron-phonon coupling on the optical properties of defects.
In particular, it implements the approach pioneered by [Alkauskas *et al.*](https://doi.org/10.1088/1367-2630/16/7/073026) to compute the lineshape function within Huang-Rhys theory.
The code interfaces with [`mace`](https://mace-docs.readthedocs.io/en/latest/) and [`phonopy`](https://phonopy.github.io/phonopy/) to evaluate the dynamical matrix and obtain the phonons of a defect-containing supercell.

_Key Features_:
 - Compute the temperature-dependent luminescence and absorption spectrum
 - Compatible with phonons computed directly with Phonopy
 - Accelerate evaluation of phonons with `mace` foundation models
 - Convenience tools to fine-tune a `mace` foundation model to your specific system

### Installation
<!-- install start -->
To install the latest version of `lineshape_tools`, create a new virtual environment and run
```
pip install lineshape_tools
```
<!-- install end -->
For more installation information and some performance considerations, see the [Installation page](https://lineshape-tools.readthedocs.io/en/latest/install.html).

### Usage 
`lineshape_tools` provides a command-line interface for interacting with the code. See
```
lineshape_tools --help
```
Detailed usage information can be found in the [Tutorials page](https://lineshape-tools.readthedocs.io/en/latest/tutorials.html).

### How to Cite
<!-- cite start -->
If you use this code, please consider citing
```bibtex
@misc{turiansky_machine_2025,
  title = {Machine Learning Phonon Spectra for Fast and Accurate Optical Lineshapes of Defects}, 
  author = {Mark E. Turiansky and John L. Lyons and Noam Bernstein},
  year = {2025},
  number = {arXiv:2508.09113},
  eprint = {2508.09113},
  archiveprefix = {arXiv},
  primaryclass = {cond-mat.mtrl-sci},
  doi = {10.48550/arXiv.2508.09113},
  url = {https://arxiv.org/abs/2508.09113}, 
}
```
<!-- cite end -->
Please also consider citing the foundational works that made this code possible on the [Citation page](https://lineshape-tools.readthedocs.io/en/latest/cite.html).

<!-- index end -->
