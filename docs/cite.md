# Citation

```{include} ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- cite start -->
   :end-before: <!-- cite end -->
```

## Relevant Literature
In addition, there are a number of works that were crucial in enabling `lineshape_tools`.
Please consider citing them in addition.

#### General lineshape theory
Consider citing the seminal work on first-principles optical lineshapes:
```bibtex
@article{alkauskas_first-principles_2014,
  title = {First-Principles Theory of the Luminescence Lineshape for the Triplet Transition in Diamond {{NV}} Centres},
  author = {Alkauskas, Audrius and Buckley, Bob B. and Awschalom, David D. and {Van de Walle}, Chris G.},
  year = {2014},
  journal = {New Journal of Physics},
  volume = {16},
  number = {7},
  pages = {073026},
  issn = {1367-2630},
  doi = {10.1088/1367-2630/16/7/073026},
  langid = {english},
}
```
Consider citing work that extends the above to include temperature dependence:
```bibtex
@article{jin_photoluminescence_2021,
  title = {Photoluminescence Spectra of Point Defects in Semiconductors: {{Validation}} of First-Principles Calculations},
  shorttitle = {Photoluminescence Spectra of Point Defects in Semiconductors},
  author = {Jin, Yu and Govoni, Marco and Wolfowicz, Gary and Sullivan, Sean E. and Heremans, F. Joseph and Awschalom, David D. and Galli, Giulia},
  year = {2021},
  month = aug,
  journal = {Physical Review Materials},
  volume = {5},
  number = {8},
  pages = {084603},
  doi = {10.1103/PhysRevMaterials.5.084603},
}
```

#### Dynamical Matrix Evaluation
If you utilize `mace` to evaluate the dynamical matrix, go to [this link](https://github.com/ACEsuit/mace?tab=readme-ov-file#references) for citation information,
and if you utilize `phonopy` to evaluate the dynamical matrix, go to [this link](https://phonopy.github.io/phonopy/citation.html) for citation information.
