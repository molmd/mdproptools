<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable-next-line MD041 -->
<div align="center" style="padding-bottom: 1em;">
<img width="400px" align="center" src="https://raw.githubusercontent.com/molmd/mdproptools/master/docs/logo.png">
</div>

# <div align="center">MDPropTools: MD Property Tools</div>

[//]: # "[![Downloads](https://static.pepy.tech/badge/mdproptools)](https://pepy.tech/project/mdproptools)"
[//]: # "[![Downloads](https://static.pepy.tech/badge/mdproptools/month)](https://pepy.tech/project/mdproptools)"
[//]: # "[![GitHub tag](https://img.shields.io/github/tag/molmd/mdproptools)](https://GitHub.com/molmd/mdproptools/tags/)"

[![codecov](https://codecov.io/gh/molmd/mdproptools/graph/badge.svg?token=K0I7FLDT6B)](https://codecov.io/gh/molmd/mdproptools)
[![PyPI - Python version](https://img.shields.io/pypi/pyversions/MDPropTools)](https://pypi.org/project/mdproptools)
[![PyPI version](https://img.shields.io/pypi/v/MDPropTools)](https://pypi.org/project/mdproptools)
[![GitHub release](https://img.shields.io/github/v/release/molmd/mdproptools)](https://pypi.org/project/mdproptools)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/molmd/mdproptools)](https://github.com/molmd/mdproptools/pulse)

`MDPropTools` is a Python package for computing structural and dynamical properties from
[LAMMPS](https://www.lammps.org/#gsc.tab=0) trajectory and output files.

The supported properties are:

| Category   | Property                                       |
| ---------- | ---------------------------------------------- |
| Structural | Radial distribution function (RDF)             |
|            | Coordination number                            |
|            | Cluster analysis                               |
|            | Hydration number                               |
|            | Number density                                 |
| Dynamical  | Mean square displacement (MSD)                 |
|            | Diffusion coefficient                          |
|            | Ionic conductivity (using Green-Kubo relation) |
|            | Viscosity (using Green-Kubo relation)          |
|            | Residence time                                 |

The release history and changelog can be found in [the changelog](CHANGELOG.md).

## 📦 Installation

`MDPropTools` can be installed using pip:

```
pip install mdproptools
```

Or by cloning the repository and running the setup script:

```
git clone https://github.com/molmd/mdproptools.git
cd mdproptools
pip install -r requirements.txt
pip install .
```

If you are planning to contribute to the development of `MDPropTools` and need access
to development tools and dependencies, you can install the package with the `dev` extra:

```
pip install ".[dev]"
```

> [!IMPORTANT]  
> `MDPropTools` has been tested using Python 3.10. It is recommended to use this version.

## 💻 Usage

For examples on how to use `MDPropTools`, please see the Jupyter notebooks in the [examples](https://github.com/molmd/mdproptools/tree/master/examples) directory.

## 👥 Contributing

All bug reports, suggestions, feedback, and pull requests occurs in the [MDPropTools GitHub repository](https://github.com/molmd/mdproptools). Some contribution guidelines can be found in the [contributing notes](CONTRIBUTING.md).

## 📖 Citation

If you use `MDPropTools` in your research, please consider citing the following paper:

- Atwi, R., Bliss, M., Makeev, M., & Rajput, N. N. (2022). [MISPR: An automated infrastructure for high-throughput DFT and MD simulations](https://www.nature.com/articles/s41598-022-20009-w). Scientific Reports, 12(1), 1-16.

## 📜 License Information

`MDPropTools` is a free, open-source software package (distributed under the [MIT license](https://github.com/molmd/mdproptools/blob/master/LICENSE)).
