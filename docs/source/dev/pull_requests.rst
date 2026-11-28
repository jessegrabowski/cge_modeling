Pull Request Step by Step
=========================

Contributions go through a fork of the `repository <https://github.com/jessegrabowski/cge_modeling>`_. You develop on
a feature branch of your fork and open a pull request against the main repository.

#. Fork the repository with the **Fork** button near the top right of its GitHub page. This creates a copy under your
   GitHub account.

#. Clone your fork and add the main repository as a remote named ``upstream``:

   .. code-block:: bash

       git clone https://github.com/<your-username>/cge_modeling.git
       cd cge_modeling
       git remote add upstream https://github.com/jessegrabowski/cge_modeling.git

#. Create a feature branch for your change. Never work on ``main``, even in your fork:

   .. code-block:: bash

       git checkout -b my-feature

#. Install the environment and the pre-commit hooks:

   .. code-block:: bash

       pixi install
       pixi run pre-commit install

   ``pixi install`` gives every operating system the same environment, from the committed ``pixi.lock``. Conda users
   can create it from ``conda_envs/environment_dev.yml`` instead.

#. Make your change, then run the checks CI will run:

   .. code-block:: bash

       pixi run test
       pixi run lint

#. Commit, then bring your branch up to date with the main repository:

   .. code-block:: bash

       git add <changed files>
       git commit
       git fetch upstream
       git rebase upstream/main

#. Push the branch to your fork:

   .. code-block:: bash

       git push -u origin my-feature

#. Open your fork's page on GitHub and click **Pull request**. Link the issue the change addresses. On a first
   contribution, a maintainer has to approve the CI runs before they start.
