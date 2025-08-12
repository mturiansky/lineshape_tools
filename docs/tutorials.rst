Tutorials
=========

The general usage and workflow of ``lineshape_tools`` is demonstrated by the following tutorials.
In general, there is a lot of overlap between steps in each tutorial.
All of the tutorials culminate in evaluating the optical lineshape for a given case and differ in the way the dynamical matrix is generated.

.. toctree::
   :maxdepth: 2

   tutorials/foundation_model.md
   tutorials/fine_tuning.md
   tutorials/phonopy.md

When utilizing ``mace`` to generate the dynamical matrix, we recommend first using the foundation model and then fine-tuning to the relaxation dataset.
(This is covered in the :doc:`first <tutorials/foundation_model>` and :doc:`second <tutorials/fine_tuning>` tutorials.)
These steps are fast and require no additional DFT calculations to be performed.
One can then compare the results and assess whether further refinement is necessary before generating more data (the optional step in the :doc:`second <tutorials/fine_tuning>` tutorial).


Python API
----------

``lineshape_tools`` was designed around the use of the command-line interface, but all of the inherent functionality is exposed in :mod:`lineshape_tools.cli`.
We recommend first familiarizing yourself with the CLI, and then, depending what functionality you need, choosing the respective function in the API.
