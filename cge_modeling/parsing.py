import re


def normalize_eq(eq: str) -> str:
    """
    Normalize an equation y = f(x) to the form y - f(x) = 0

    Parameters
    ----------
    eq: str
        A string representing an equation

    Returns
    -------
    normalized_eq: str
        A string representing the same equation in normalized form
    """
    equals_pattern = "(?<!axis)="
    if re.search(equals_pattern, eq) is None:
        return eq
    lhs, rhs = re.split(equals_pattern, eq, maxsplit=1)
    return f"{lhs} - ({rhs})"


def wrap_in_ravel(eq: str) -> str:
    return f"({eq}).ravel()"
