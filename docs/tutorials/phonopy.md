# From Phonopy
This tutorial describes how to evaluate the optical lineshape using force constants produced by phonopy.
The most obvious use for this functionality is to compute the lineshape function using any of the [DFT codes supported by phonopy](https://phonopy.github.io/phonopy/interfaces.html).

## Step 0: Identify and Calculate a Transition
The first step is to identify a defect candidate and a potential optical transition.
For a transition, there will be a ground state and an excited state, and we perform an atomic relaxation in each state to determine the respective equilibrium geometries.

This step occurs indepedent of `lineshape_tools` and is left as an exercise for the reader.

## Step 1: Obtain `FORCE_CONSTANTS` File Using Phonopy
This step is mostly independent of `lineshape_tools`, so we refer the reader to the [phonopy documentation](https://phonopy.github.io/phonopy/index.html) and also the comprehensive [examples subdirectory](https://github.com/phonopy/phonopy/tree/develop/example) in the phonopy repository.
The main goal is to obtain the `FORCE_CONSTANTS` file, which is accomplished by adding the `--writefc` tag to a call that parses the `FORCE_SETS` file.
This file can be parsed and converted by `lineshape_tools` using the command:
```sh
lineshape_tools convert-from-phonopy /path/to/FORCE_CONSTANTS /path/to/equilibrium/structure
```

For example using `VASP` as the calculator as of 2025-08-01, the following sequence of commands should work:
```sh
# produce displaced POSCAR-* structures
phonopy -d --dim 1 1 1 -c CONTCAR

# perform VASP calculations on each POSCAR-* file and produce corresponding vasprun-*.xml
# these files are then used to create the FORCE_SETS file with the following command
phonopy -f vasprun-*.xml

# convert to FORCE_CONSTANTS
phonopy-load --writefc

# convert to dynmat.npz
lineshape_tools convert-from-phonopy FORCE_CONSTANTS CONTCAR
```

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
