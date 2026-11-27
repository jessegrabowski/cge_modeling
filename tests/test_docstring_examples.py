import importlib
import inspect
import pkgutil
import re
import textwrap

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import cge_modeling

API_PAGES = Path(__file__).parent.parent / "docs" / "source" / "api"
EXAMPLES_SECTION = re.compile(
    r"^Examples\n-{8,}\n(?P<body>.*?)(?=^\S[^\n]*\n-{3,}\n|\Z)", re.M | re.S
)
CODE_BLOCK = re.compile(r"^\.\. code-block:: python\n\n(?P<code>(?:(?: {4}.*)?\n)+)", re.M)


def public_members(module):
    for name, member in vars(module).items():
        if name.startswith("_") or getattr(member, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(member):
            yield f"{module.__name__}.{name}", member
        elif inspect.isclass(member):
            yield f"{module.__name__}.{name}", member
            for attribute_name, attribute in vars(member).items():
                if attribute_name.startswith("_"):
                    continue
                if isinstance(attribute, property | staticmethod | classmethod):
                    attribute = (
                        attribute.fget if isinstance(attribute, property) else attribute.__func__
                    )
                if inspect.isfunction(attribute):
                    yield f"{module.__name__}.{name}.{attribute_name}", attribute


def docstring_examples():
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(cge_modeling.__path__, prefix="cge_modeling.")
    ]
    for module in modules:
        for qualified_name, member in public_members(module):
            # getdoc strips the trailing newline, which the last line of a final code block needs to match.
            docstring = (inspect.getdoc(member) or "") + "\n"
            section = EXAMPLES_SECTION.search(docstring)
            if section is None:
                continue
            for index, block in enumerate(CODE_BLOCK.finditer(section["body"])):
                yield pytest.param(textwrap.dedent(block["code"]), id=f"{qualified_name}[{index}]")


@pytest.mark.parametrize("code", list(docstring_examples()))
def test_docstring_example_runs(code):
    try:
        exec(compile(code, "<docstring example>", "exec"), {"__name__": "__main__"})
    finally:
        plt.close("all")


def documented_api():
    """Yield the defining module and name of every object an autosummary on the API pages lists."""
    for page in sorted(API_PAGES.glob("*.rst")):
        module = None
        for line in page.read_text().splitlines():
            if line.startswith(".. currentmodule::"):
                module = importlib.import_module(line.split("::")[1].strip())
            elif line.startswith("    ") and module and not line.strip().startswith(":"):
                member = getattr(module, line.strip())
                yield f"{member.__module__}.{member.__name__}"


def test_every_documented_api_object_has_an_example():
    documented = set(documented_api())
    examples = {example.id.rsplit("[", 1)[0] for example in docstring_examples()}

    assert len(documented) > 30
    assert documented <= examples, sorted(documented - examples)
