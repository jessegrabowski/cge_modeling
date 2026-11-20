from typing import Any, cast

from IPython import display


def print_latex(eqs):
    for eq in eqs:
        display(eq)


def union_many(x, *args):
    for y in args:
        if not isinstance(y, list | tuple | set):
            y = (y,)
        x = x.union(set(y))
    return x


def brackets_to_snake_case(name):
    """
    Rename a variable x[i, j] to x_i_j
    """
    name = name.replace("]", "")
    name = name.replace("[", "_")
    name = name.replace(",", "_")
    name = name.replace(" ", "")
    return name


def at_least_list(x: Any) -> list[Any]:
    """
    Wrap non-list objects in a list

    Parameters
    ----------
    x: Any
        An object to wrap in a list if it is not already a list

    Returns
    -------
    x: list
        The input object wrapped in a list if it was not already a list
    """
    if isinstance(x, list):
        return x
    else:
        return cast(list, [x])
