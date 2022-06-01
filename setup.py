import os

import setuptools

module_dir = os.path.dirname(os.path.abspath(__file__))

setuptools.setup(
    name="mdproptools",
    version="0.0.3",
    author="Rasha Atwi, Matthew Bliss, Maxim Makeev",
    author_email="rasha.atwi@stonybrook.edu, matthew.bliss@stonybrook.edu, "
                 "maxim.makeev@stonybrook.edu",
    description="mdproptools contains MD analysis tools",
    long_description=open(os.path.join(module_dir, "README.md")).read(),
    url="https://github.com/molmd/mdproptools",
    install_requires=["numpy >= 1.21.1",
                      "matplotlib >= 3.3.1",
                      "scipy >= 1.5.2",
                      "pandas >= 1.1.2",
                      "seaborn >= 0.11.0",
                      "statsmodels >= 0.12.0",
                      "numba == 0.53.1",
                      "llvmlite == 0.36.0"
                      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    package_data={},
)