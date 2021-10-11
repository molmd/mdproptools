import os

import setuptools

module_dir = os.path.dirname(os.path.abspath(__file__))

setuptools.setup(
    name="mdproptools",
    version="0.0.2",
    author="Rasha Atwi, Matthew Bliss, Maxim Makeev",
    author_email="rasha.atwi@stonybrook.edu, matthew.bliss@stonybrook.edu, "
                 "maxim.makeev@stonybrook.edu",
    description="mdproptools contains MD analysis tools",
    long_description=open(os.path.join(module_dir, "README.md")).read(),
    url="https://github.com/tufts-university-rajput-lab/mdproptools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status ::  2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    package_data={},
)