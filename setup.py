import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='microtubule_catastrophe_pkg',
    version='0.0.1',
    author='Amanda Li, Cindy Cao, Maggie Sui',
    author_email='amandali@caltech.edu, scao@caltech.edu, msui@caltech.edu',
    description='Utilities for analyzing microtubule catastrophe data',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)