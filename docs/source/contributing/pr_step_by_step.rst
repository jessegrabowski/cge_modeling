Pull request step-by-step
=============
The preferred workflow for contributing to `cge_modeling` is to fork the GitHub [repository](https://github.com/jessegrabowski/cge_modeling), clone it to your local machine, and develop on a feature branch.

Steps
------------
1. Fork the project repository by clicking on the ‘Fork’ button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the `cge_modeling` [repository](https://github.com/jessegrabowski/cge_modeling) from your GitHub account to your local disk, and add the base repository as a remote::

    git clone https://github.com/jessegrabowski/cge_modeling.git
    cd cge_modeling
    git remote add upstream https://github.com/jessegrabowski/cge_modeling.git

3. Create a feature branch to hold your development changes::
    git checkout -b my-feature

.. warning::
    Always use a `feature` branch. It’s good practice to never routinely work on the main branch of any repository.

The easiest (and recommended) way to set up a development environment is via miniconda:
First set the directory to the repository path::
    cd "put_here_the_path"


.. tabs::

   .. tab:: Linux/MacOS

      .. code-block:: Linux/MacOS

         conda env create -f conda-conda_envs/environment.yml

   .. tab:: Windows

      .. code-block:: Windows

         conda env create -f conda_envs\environment.yml

   .. tab:: Windows (Git Bash)

      .. code-block:: conda env create -f conda-conda_envs\environment.yml




The activate the environment::

    conda activate cge-modeling
    pip install -e .

Alternatively you may (probably in a virtual environment) run::

    pip install -e .
    pip install -r requirements-dev.txt

Develop the feature on your feature branch::

    git checkout my-feature   # no -b flag because the branch is already created

Before committing, run pre-commit checks::

    pip install pre-commit
    pre-commit run --all      # 👈 to run it manually
    pre-commit install        # 👈 to run it automatically before each commit

Add changed files using git add and then git commit files::

    $ git add modified_files
    $ git commit

to record your changes locally.

After committing, it is a good idea to sync with the base repository in case there have been any changes::

    git fetch upstream
    git rebase upstream/main

Then push the changes to the fork in your GitHub account with::

    git push -u origin my-feature
    If this is your first contribution, the start of some CI jobs will have to be approved by a maintainer.

Go to the GitHub web page of your fork of the `cge_modeling` repo. Click the ‘Pull request’ button to send your changes to the project’s maintainers for review. This will send a notification to the committers.
