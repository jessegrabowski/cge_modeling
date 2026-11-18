Installation
============


pip
***
``cge_modeling`` is on PyPI. Install `pytensor <https://pytensor.readthedocs.io/en/latest/>`_ from conda-forge first,
then ``cge_modeling`` with pip:

.. code-block:: bash

    conda create -n cge-modeling -c conda-forge python=3.12 pytensor
    conda activate cge-modeling
    pip install cge_modeling

Installing straight from PyPI into a plain virtual environment also works, but pytensor then needs a C compiler on the
system path to compile models. The conda-forge build of pytensor comes with a C compiler and a BLAS library already
configured.


Development installation
************************
The repository is a `pixi <https://pixi.sh>`_ workspace. From a clone, ``pixi install`` creates an environment from
the committed lock file, holding the runtime, test, and documentation dependencies plus ``cge_modeling`` in editable
mode. ``pixi shell`` activates it:

.. code-block:: bash

    git clone https://github.com/jessegrabowski/cge_modeling.git
    cd cge_modeling
    pixi install
    pixi shell

The workspace defines tasks for the common jobs, which run without activating a shell:

.. code-block:: bash

    pixi run test                # the test suite, extra pytest arguments pass through
    pixi run lint                # every pre-commit hook on every file
    pixi run docs-build          # the documentation, into docs/build/html

For conda directly, ``conda_envs/environment_dev.yml`` is the same environment exported from the pixi workspace, and
its pip section installs ``cge_modeling`` in editable mode:

.. code-block:: bash

    conda env create -f conda_envs/environment_dev.yml
    conda activate cge-dev
