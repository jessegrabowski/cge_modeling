Style Guide
===========

``ruff`` settles anything mechanical (formatting, import order, line length), and CI runs it, so
``pixi run lint`` decides those questions before review does. This page covers the rest: the judgment the tools
cannot make.

Most of it you will get right by writing code that looks like the file around it.

Code
----

Keep hot paths lean: no redundant checks, no invariant recomputed inside a loop that could be hoisted out of it,
no silent ``O(n^2)`` where ``O(n)`` works. A solve calls the compiled residual and Jacobian functions at every
iteration, and a simulation repeats the solve at every step, so anything on that path is paid for many times over.

Errors should be specific and loud. A bare ``except:``, or an ``except Exception: pass`` that swallows a failure,
turns a bug into a mystery three layers away. Validate input where a wrong value would otherwise surface as a
confusing error deep in a graph or a solver, and nowhere else. A check that cannot fire is noise.

Where you do raise, name the fix in the message. It reaches the reader at the moment they need it, which no amount
of documentation does:

.. code-block:: python

    raise ValueError(
        "The prior must be finite. Fill missing cells with 0.0 before building a SymbolicSAM"
    )

Design
------

Write the simplest thing that works. Generality for a future that has not arrived is a cost paid now against a
benefit that may never come. Two pieces of code that look alike today but change for different reasons are not
duplication, and forcing them together couples them wrongly, so the rule of three is a good prior before
extracting a helper.

Keep one function at one altitude: raw string-mangling next to high-level orchestration means a helper is missing.
Be consistent about failure, too. One module should not raise for some errors, return ``None`` for others, and
return a sentinel for a third kind of the same failure.

The usual anti-patterns to watch for:

* A boolean flag that makes one function do two things, and the ``do_thing(True, False)`` call site it leads to.
* Parameter lists too long to hold in your head.
* A dict standing in for an object, where a dataclass would say what the fields are.
* Mutable default arguments, and hidden global state.
* A function that both computes and mutates.

Naming and shape
----------------

A name should say what something is for, so the code reads as its own documentation. Avoid names that describe a
type rather than a role: ``data``, ``tmp``, ``obj``, ``result2``. Single letters are fine where they are the
mathematical convention (``i``, ``x``, an elasticity's ``sigma``) and nowhere else. Name the same concept the same
way the surrounding code does, and name the constants: a bare ``0.9`` in an update tells the reader nothing, where
``damping`` tells them what it is for.

Shape carries meaning too. Prefer guard clauses to nesting, so the happy path stays prominent and error handling
sits at the edges. Break up expression soup with named intermediates, since the name *is* the documentation and
costs one line. When two branches do the same kind of thing, shape them the same way, so that an asymmetry signals
a real difference rather than drift. Within a module, read top down: public API first, helpers below it, reading
order roughly matching call order.

Comments
--------

Comments explain the **why**, and the code already says what it does. Fewer is better. Every comment can drift out
of sync, so it has to change a reader's understanding to earn its place. A better name usually beats a comment.

Do not commit:

* Comments that narrate the code (``# increment the counter``).
* Commented-out code. That is what version control is for.
* Process notes: ``# previously we used a loop here``, ``# fix for #123``, or a ``TODO`` with no owner and no
  context. Those belong in the commit message.

Reflow comment prose to the full line width. Short broken-up lines waste vertical space for no gain.

Docstrings
----------

Docstrings are numpydoc, and they document the **current contract**. Write each one as if the function appeared in
the codebase fresh today. A reader who cloned the repo an hour ago should never meet a sentence that only makes
sense if they know what the code used to do.

That rules out, however technical the prose sounds: explaining current behavior by contrasting it with a previous
version; references to audits, benchmarks, pull requests or incidents; and "Notes" sections that exist to justify
a recent change rather than to document an invariant. The *why we changed it* goes in the commit message, where it
stays findable without being in front of everyone who hits ``?`` in a REPL.

The rest of the rules:

* **Active voice.** "Compute the gradient", not "The gradient is computed".
* **Every parameter gets a human-readable type**: ``list of int``, not ``list[int]``. Describe a genuinely nested
  type in prose rather than pasting a type hint into the docstring.
* **Optional arguments say so** (``, optional`` on the type line), and the **default goes in the last sentence** of
  the description, not on the type line.
* **Return values are named**, even when nothing ever binds them.
* **No Raises sections.** The error message is the documentation.
* **No module-level docstrings.** If a module's purpose is not evident from its name and contents, the fix is a
  better name or a split.
* **Math goes in** ``.. math::`` **directives**, never as backticked ASCII. Use a raw string so the backslashes
  survive. Inline, a mathematical symbol takes a ``:math:`` role and a code identifier takes double backticks: "the
  elasticity :math:`\sigma` (parameter ``sigma``)".
* Cross-reference with Sphinx roles: ``:func:``, ``:class:``, ``:meth:``, ``:mod:``.

A short private helper whose name and signature already say everything can go without a docstring. What it cannot
have is a chatty paragraph standing in for a short structured one:

.. code-block:: python

    def index_by_parameter(tree: xr.DataTree, parameter: str) -> xr.DataTree:
        """
        Index a simulation group by the values of one scalar parameter instead of by ``step``.

        Parameters
        ----------
        tree : xr.DataTree
            A group of a ``CGEModel.simulate`` result, such as ``result["euler"]``, whose ``variables`` and
            ``parameters`` children are indexed by ``step``.
        parameter : str
            Name of a scalar parameter whose value changes strictly monotonically along ``step``.

        Returns
        -------
        indexed : xr.DataTree
            The same tree with ``parameter`` as the dimension of both children.
        """

Examples
--------

Every public entrypoint carries an ``Examples`` section. Three rules:

#. **A lead-in sentence, always**, even when the code looks self-evident. Never open the section with the
   directive. Where two entrypoints have nearly identical code, the lead-in is the only thing telling them apart,
   so it says what to reach for this one for.
#. ``.. code-block:: python``, **never the** ``>>>`` **prompt.** A prompt cannot be pasted into a script.
#. **Complete and runnable on its own**, imports included, small and tight. Add a second block, with its own
   lead-in, when there is a real fork in usage or a sharp edge worth showing.

.. code-block:: rst

    Examples
    --------
    Balance a three-account circular flow whose prior payments disagree, so that every account's receipts equal
    its payments:

    .. code-block:: python

        import pandas as pd

        import cge_modeling as cm

        accounts = ["Labor", "Firms", "Households"]
        prior = pd.DataFrame([[0.0, 100.0, 0.0],
                              [0.0, 0.0, 120.0],
                              [90.0, 0.0, 0.0]],
                             index=accounts,
                             columns=accounts)

        estimate = cm.SymbolicSAM(prior).fit()
        estimate.sam

Modern Python
-------------

The supported floor is Python 3.12, so write for it:

* PEP 604 unions (``int | None``) and PEP 585 generics (``list[int]``), not ``Optional`` or ``typing.List``.
* **No** ``from __future__ import annotations``. It is unnecessary here.
* f-strings, context managers, ``pathlib`` over ``os.path`` string-mangling, ``enumerate`` and ``zip`` over index
  bookkeeping.
* Comprehensions where they read more clearly than an accumulator loop, and not where they get dense enough to
  obscure what is happening.
* **Imports at the top of the module.** A function-local import is warranted only to break a genuine circular
  dependency or to guard an optional dependency, and both cases are obvious from context.
* Public functions and methods carry type hints. They are read as documentation.

Tests
-----

Test code is code, and everything above applies to it, with one deliberate exception. **Duplicated setup in tests
is usually worth keeping.** A test earns its value by being auditable as one self-contained block: what was built,
solved, patched and asserted, all visible without jumping to a fixture defined four hundred lines away. Repeated
arrange-phase boilerplate is a smaller cost than a fragmented test, even at six or eight occurrences.

Extract a fixture when the block is long enough to bury the assertion it exists to set up, when it encodes an
invariant that must stay identical across tests, or when a signature change would otherwise mean editing it
everywhere.

Commit messages
---------------

Subject line in the imperative, short enough to read at a glance, naming the thing that changed: "Add a scenario
dimension to simulate results" beats "Update simulate".

**Never hard-wrap the body.** One paragraph is one line, however long. Manual line breaks at 72 characters become
unreadable the moment anything reflows them, and they fossilize a width nothing actually uses. Blank lines separate
paragraphs, and deliberately structured content (lists, tables, aligned columns) keeps its own line breaks.

A body is worth writing when the *why* is invisible in the diff. It is not worth writing to restate the diff in
prose.

Cruft
-----

The diff should look like it was written by someone who cleaned up after themselves: no leftover ``print``
statements, no dead code, no unused imports or locals, no stray scratch files. A file that genuinely should not be
tracked belongs in ``.gitignore``, not deleted and recreated.

Every dependency is a liability. Question a new one that the standard library, NumPy, xarray or PyTensor already
covers.
