# Rate Constant Matrix Contraction Method for Stiff Master Equations with Detailed Balance
Code for numerical experiments in the paper

> S. Iwata, T. Oki, and S. Sakaue, "Rate constant matrix contraction method for stiff master equations with detailed balance", SIAM Journal on Scientific Computing, 48(1)A261–A285, 2026.

## Folder structure
- `cpp`: C++ code for experiments
- `data`: data files
- `result`: experiment results will be saved here
- `python`: Python code for visualizing the results
- `experiment.ipynb`: Jupyter notebook for reproducing the experiments

## Running the C++ code
### Requirements
- C++ compiler with C++17 support (tested with Apple Clang 14.0.3)
- [CMake](https://cmake.org/) (version >= 3.20)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) for Linux and macOS and [pkg-config-lite](https://sourceforge.net/projects/pkgconfiglite/) for Windows
- [Boost](https://www.boost.org/) (version >= 1.7)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (version >= 3.4)
- [GMP](https://gmplib.org/) (tested with version 6.3.0)

### Compile
```sh
cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release
make -j -C build
```

### Run
In the following, `<DATA>`, `<TYPE>`, `<REFERENCE>`, and `<PRECISION>` are placeholders for the following values:

- `<DATA>`: name of the data file in the `data` folder (`synthetic`, `DFG`, or `WL`)
- `<TYPE>`: type of the RCMC method (`A` or `B`)
- `<REFERENCE>`: method to compute the reference time (`diag`, `eigen`, or `gershgorin`)
- `<PRECISION>`: floating-number precision to be used (`15` (= double precision), `50`, `100`, `200`, or `400`)

#### Run the RCMC method
```sh
build/bin/pop <DATA> -t <TYPE> --time <REFERENCE> -p <PRECISION>
```

The default precision is `15`.
Approximate solutions $q^{(0)}, \dotsc, q^{(n)}$ will be saved in `result/<DATA>-pop-<TYPE>-<PRECISION>.txt`.
Reference times $t^{(0)}, \dotsc, t^{(n)}$ are also computed to measure the running time, but will not saved. To save the reference times, use `build/bin/epoch` below.

#### Computing reference times
```sh
build/bin/epoch <DATA> -p <PRECISION>
```

The default precision is `15`.
All of the three methods (`diag`, `eigen`, and `gershgorin`) are run.
The results will be saved in `result/<DATA>-epoch-<PRECISION>.txt`.

#### Eigendecomposition of rate constant matrices
```sh
build/bin/eigen <DATA> -p <PRECISION>
```

The default precision is `200`.
The results will be saved in `result/<DATA>-eigen-<PRECISION>.txt`.

#### Computing the curve of analytical solutions
Before running the following, compute the eigendecomposition (using `build/bin/eigen` above) with the same precision.

```sh
build/bin/ode <DATA> -p <PRECISION>
```

The default precision is `200`.
The results will be saved in `result/<DATA>-ode-<PRECISION>.txt`.

#### Computing analytical solutions at reference times
Before running the following, compute the reference times (using `build/bin/epoch`) with double precision and eigendecomposition (using `build/bin/eigen` above) with the same precisions.

```sh
build/bin/ode <DATA> --epoch -p <PRECISION>
```

The default precision is `200`.
The results will be saved in `result/<DATA>-ode-epoch-<PRECISION>-15.txt`.

## Visualizing the results
### Requirements
- [Python](https://www.python.org/) (version >= 3.9)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [gmpy2](https://gmpy2.readthedocs.io/en/latest/)

The results can be visualized using the following functions in `python/plot.py`: `show_rcmc`, `show_ode`, `show_rcmc_ode`, `show_rcmc_ode_all_references`, `show_pi_error_log`, `show_pi_error_semilog`, `show_pi_error_all_references`, `show_inf_error_semilog`, and `stack_semilog_plots`.
For example, the following code (executed in Jupyter notebook) plots the output of the RCMC method of Type A with "diag" method applied to the synthetic data.

```py
from python.plot import show_rcmc
show_rcmc('synthetic', 'A', 'diag')
```

Usage of other functions is partially shown in `experiment.ipynb`.
