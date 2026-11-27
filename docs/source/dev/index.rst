Contributing
============

The repository is a `pixi <https://pixi.sh>`_ workspace, and every check that CI runs is a pixi task you can run
locally with the same command.

**Every pull request needs an issue first.** Open one describing what you mean to change, and wait for a reply
before writing the code. The design may already be settled, the behavior may be deliberate, or someone may be
halfway through the same work. Link the issue from the pull request.

Environment
-----------

.. code-block:: bash

    git clone https://github.com/jessegrabowski/cge_modeling.git
    cd cge_modeling
    pixi install
    pixi shell

``pixi install`` resolves the runtime, test and documentation dependencies from the committed ``pixi.lock`` and
installs cge_modeling in editable mode. ``conda_envs/environment_dev.yml`` is the same environment exported for conda
users. A pre-commit hook generates it, so do not edit it by hand.

Tasks
-----

.. code-block:: bash

    pixi run test                    # the test suite; extra pytest arguments pass through
    pixi run test tests/base         # one directory
    pixi run lint                    # every pre-commit hook on every file
    pixi run -e docs docs-build      # build the documentation into docs/build/html
    pixi run -e docs docs-serve      # build, then serve it at http://127.0.0.1:8000

CI runs the test suite in subsets across Linux and Windows and runs the pre-commit hooks. Read the Docs builds the
documentation for every pull request. A pull request has to pass all of them.

Code style
----------

Pre-commit with ruff is authoritative. Run ``pixi run lint`` before proposing a change. For the conventions no tool
enforces, see the :doc:`style guide <style_guide>`. Docstrings are NumPy style with human-readable types, and the
documentation build treats every warning and every unresolved cross-reference as an error, so a docstring that
names a type or a function that does not exist fails the build.

Tests
-----

Tests live under ``tests/`` and mirror the package layout. Compile PyTensor functions once per test and sweep many
inputs through them, because repeated compilation dominates the run time. Prefer parametrization over conditionals
inside a test, and extend an existing test when the new coverage belongs with it.

The Linux jobs in ``.github/workflows/run_tests.yml`` split the suite into subsets that must run every test exactly
once. A pre-commit hook checks this, and fails when a test runs in no Linux subset or in two, or when a subset names
a test that does not exist. A new file under ``tests/`` lands in the catch-all subset on its own. Moving or renaming
a test that a subset names by node id means updating that subset in the same pull request.

Documentation
-------------

The API reference under ``docs/source/api/`` is a set of hand-written ``autosummary`` pages, one per module. A new
public class or function appears in the reference only after it is added to the page for its module.

Example notebooks live under ``notebooks/``, which the documentation reads as ``docs/source/examples/``. Execute them
top to bottom before you commit them, because the build does not execute them and a pre-commit hook rejects cells
run out of order. The gallery shows every notebook in the folders listed in ``folder_title_map`` in
``sphinxext/generate_gallery.py``, so a new folder needs an entry there.

Working with LLMs
-----------------

LLM-assisted contributions are welcome, on two conditions.

**Disclose it.** Say in the pull request that a model was involved, and roughly how much. A tab-completion here and
there is not the same as a generated module, and a reviewer reads the two differently.

**You are responsible for every line you submit, not the model.** You understand what the code does and why, you
can defend each decision in review, and you have run it. Review comments come to you, and "the model wrote it that
way" answers nothing. Code that nobody understands costs more to maintain than no contribution at all, and a pull
request whose author cannot explain it will be closed.

.. toctree::
   :hidden:

   style_guide
