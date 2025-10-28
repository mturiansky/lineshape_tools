# From a Foundation Model
This tutorial describes how to evaluate the optical lineshape using a `mace` foundation model.

## Step 0: Identify and Calculate a Transition
The first step is to identify a defect candidate and a potential optical transition.
For a transition, there will be a ground state and an excited state, and we perform an atomic relaxation in each state to determine the respective equilibrium geometries.

This step occurs indepedent of `lineshape_tools` and is left as an exercise for the reader.

## Step 1: Compute the Dynamical Matrix
To compute the dynamical matrix, we must specify an atomic geometry as a starting point.
The atomic geometry should correspond to the "final state" of the transition: for luminescence, this is the ground state, and for absorption, this is the excited state.
This geometry is just a starting point, as we will perform an atomic relaxation to determine the foundation model's prediction for the equilibrium geometry.
(The relaxation can be skipped with the `--no-relax-struct` tag, but we recommend against this.)
Ultimately, since the current foundation models have no notion of electronic structure, they will predict the same geometry for the ground and excited states.

To obtain the dynamical matrix, run
```sh
lineshape_tools compute-dynmat /path/to/structure
```
Note that the structure is read using the [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function, so a wide variety of first-principles codes are supported.
The default assumes you are running on a GPU.
If you are limited to CPUs, append `--device cpu` to the command.

If the above command fails with the error
```sh
RuntimeError: Model download failed and no local model found
```
then you will need to download a foundation model manually.
Navigate to the [foundation model release page](https://github.com/ACEsuit/mace-foundations/releases) and download one of the `*.model` files.
For example,
```sh
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
```
The dynamical matrix can be obtained by specifying the path to the model, for example,
```sh
lineshape_tools compute-dynmat /path/to/structure --mace-model ./mace-omat-0-medium.model
```

If the above suceeded, you should have produced a file `dynmat.npz` in the current directory.

## Step 2: Evaluate the Optical Lineshape
Finally, we evalute the luminescence intensity by issuing the following command
```sh
lineshape_tools compute-lineshape /path/to/ground/structure /path/to/excited/structure dynmat.npz
```
This should produce a file `lineshape.txt` that contains the frequency grid in the first column, the phonon density of states in the second, the spectral density in the third, and the luminescence intensity in the fourth.
We can then plot the results with the plotting program of our choice.
Alternatively, we can append the `--plot subplots` tag to produce plots of these three quantities.
We note that broadening parameters are highly system specific, and the default values will likely need to be tuned.
This can be accomplished using the various tags listed in the `lineshape_tools compute-lineshape --help` text.
If the absorption lineshape is desired instead, append the `--absorption` tag.

```{tip}
If your structure files contain total energies (e.g., `vasprun.xml` files), then the energy difference can be determined automatically.
However, this will not include corrections for charge or other effects.
Alternatively, the energy difference can be provided with the `--de` tag.
```
