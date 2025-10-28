# From a Fine-Tuned Model
This tutorial describes how to evaluate the optical lineshape by fine-tuning a `mace` foundation model to defect-specific training data.
With this method, accuracy rivaling that of the underlying DFT calculations can be obtained.

## Step 0: Identify and Calculate a Transition
The first step is to identify a defect candidate and a potential optical transition.
For a transition, there will be a ground state and an excited state, and we perform an atomic relaxation in each state to determine the respective equilibrium geometries.

```{important}
As we will be fine-tuning our model, it is useful to keep all atomic relaxation data. Including more data will increase the accuracy of our model, and this relaxation data is "free", as it needed to be obtained to determine the equilibrium geometries anyway.
```

This step occurs indepedent of `lineshape_tools` and is left as an exercise for the reader.

## Step 1: (*Optional*) Generate Additional Training Data
For some cases, it may be desirable to generate additional data for training to improve the accuracy of the fine-tuned model.
This can be accomplished by the issuing the following command
```sh
lineshape_tools gen-confs /path/to/structure num_confs
```
where the structure corresponds to the "final state" of the transition (see [Step 4](#step4)) and `num_confs` is the number of additional configurations desired.
The above command will produce a directory `conf` with `num_confs` subdirectories, each containing a `VASP`-style `POSCAR` file.
```{tip}
Different structure file formats can be obtained with the `ase convert` command. For example, `ase convert POSCAR struct.xyz`
```
The default is to generate random displacements, and we recommend 10 configurations as a reasonable starting point.
There is a small advantage to using the `--strategy phon_opt` tag, but the difference is relatively minor compared to the default random configurations.
See [our paper](https://doi.org/10.48550/arXiv.2508.09113) for more details.

## Step 2: Collect Data
To fine-tune the `mace` foundation model, we need to generate a database of configurations that have energies and forces evaluated.
We will collect all of this information into a single file using the `lineshape_tools collect` command.

```{note}
The files will be read with [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read), but not all files accepted by this function contain the necessary information. For example in VASP, the vasprun.xml file should be used instead of the CONTCAR file.
```

Potential sources of additional data:
1. Defect supercell atomic relaxation
2. Pristine supercell atomic relaxation
3. Additional data generated in Step 1

As an example, assuming `defect.vasprun.xml` contains information on the defect atomic relaxation and `supercell.vasprun.xml` contains information on the pristine supercell relaxation, then we can collect this data with the following commands
```sh
lineshape_tools collect defect.vasprun.xml supercell.vasprun.xml
```
If additional data was generated in step 1, then
```sh
lineshape_tools collect confs/*/vasprun.xml
```
will collect that data and append it to the database (assuming `VASP` was used to perform the DFT calculations).

```{note}
Each successive call will append to `database.extxyz` in the current directory if found.
```

```{important}
Do not mix calculations from the ground and excited state (or more generally, calculations with different electronic sturcture but the same atoms). This will confuse the model as the same configurations may have different energies and forces in the database.
```

If the above succeeded, we should have produced a file `database.extxyz`.

## Step 3: Model Training
To set up the training input files, run
```sh
lineshape_tools gen-ft-config
```
This will produce a file `default.config`, which contains sensible default training parameters.
It also provides paths to the `database.extxyz` file generated in the previous step and the foundation model that is being used.
We recommend looking over `default.config`, as there are a number of parameters that could potentially be updated.

One parameter in particular that needs to be addressed in `default.config` is `E0s`.
`E0s` should be a dictionary that maps an atomic number to the energy of the isolated atom in a box.
For example for the NV center in diamond, our defect cell contains C and N atoms.
We run two DFT calculations, each containing an isolated C and N atom, respectively.
These DFT calculations are run at the same level of theory as the defect supercell calculation.
We then update `config.default` to read
```sh
E0s: "{6: X.XXXX, 7: Y.YYYY}"
```
where the total energies from the isolated atom calculations are inserted.

Alternatively, an estimation method is included that can be activated with the `--estimate-e0s` tag.
The method requires evaluating the model for each configuration in the database, so we recommend performing this step on a GPU if utilized.

To actually perform the training, run
```sh
mace_run_train --config config.default --device cuda --enable_cueq True
```
This command assumes you're training on a GPU (highly recommended) and have cuEquivarience installed (also recommend, see the [Installation page](../install.md)).

If the above succeeded, you should have produced a file `fine-tuned.model` (amongst other files).

(step4)=
## Step 4: Compute the Dynamical Matrix
To compute the dynamical matrix, we must specify an atomic geometry as a starting point.
The atomic geometry should correspond to the "final state" of the transition: for luminescence, this is the ground state, and for absorption, this is the excited state.
This geometry is just a starting point, as we will perform an atomic relaxation to determine the foundation model's prediction for the equilibrium geometry.

The relaxation can be skipped with the `--no-relax-struct` tag.
Indeed, an atomic relaxation shouldn't be needed if the training produced a model that effectively learned the DFT equilibrium structure.
In practice, there is no harm in leaving the relaxation on.

To obtain the dynamical matrix, run
```sh
lineshape_tools compute-dynmat /path/to/structure --mace-model ./fine-tuned.model
```
Note that the structure is read using the [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function, so a wide variety of first-principles codes are supported.
The default assumes you are running on a GPU.
If you are limited to CPUs, append `--device cpu` to the command.

If the above suceeded, you should have produced a file `dynmat.npz` in the current directory.

## Step 5: Evaluate the Optical Lineshape
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
