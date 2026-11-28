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
