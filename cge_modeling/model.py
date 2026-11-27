from typing import TYPE_CHECKING, Any

from cge_modeling.base.context import ModelObjectVariable, XTensorEquation, modelcontext

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel


def Variable(
    name: str,
    dims: str | list[str] | tuple[str, ...] | None = None,
    *,
    description: str | None = None,
    latex_name: str | None = None,
    extend_subscript: bool | int = False,
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    r"""
    Create an endogenous variable, which a solver finds the value of, and register it on a model.

    Parameters
    ----------
    name : str
        Name of the variable, unique among the model's variables and parameters.
    dims : str or sequence of str, optional
        Dimensions the variable is indexed by, each a key of the model's coords. Default is None, which makes a scalar.
    description : str, optional
        Description shown in :meth:`~cge_modeling.base.cge.CGEModel.summary`. Default is None.
    latex_name : str, optional
        LaTeX symbol used in tables, such as ``"alpha"``, which renders as :math:`\alpha`. Default is None, which uses
        ``name``.
    extend_subscript : bool or int, optional
        Move the last underscore-separated tokens of the name into the index subscript, so that ``"P_VA"`` indexed by
        ``i`` renders as :math:`P_{\mathrm{VA}, i}`. True moves one token, and an int moves that many. Default is
        False.
    model : CGEModel, optional
        Model to register the variable on. Default is None, which uses the model of the enclosing ``with`` block.

    Returns
    -------
    variable : XTensorVariable
        The variable as a symbolic xtensor, to build equations from.

    Examples
    --------
    Declare output indexed by sector, with a description for the model summary:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector", description="Output of each sector")

        model.variable_names  # ['Y']
    """
    return modelcontext(model)._register_model_object(
        name=name,
        dims=dims,
        kind="variable",
        description=description,
        latex_name=latex_name,
        extend_subscript=extend_subscript,
    )


def Parameter(
    name: str,
    dims: str | list[str] | tuple[str, ...] | None = None,
    *,
    description: str | None = None,
    latex_name: str | None = None,
    extend_subscript: bool | int = False,
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    r"""
    Create a parameter, which stays fixed while a solver runs, and register it on a model.

    Parameters
    ----------
    name : str
        Name of the parameter, unique among the model's variables and parameters.
    dims : str or sequence of str, optional
        Dimensions the parameter is indexed by, each a key of the model's coords. Default is None, which makes a scalar.
    description : str, optional
        Description shown in :meth:`~cge_modeling.base.cge.CGEModel.summary`. Default is None.
    latex_name : str, optional
        LaTeX symbol used in tables, such as ``"alpha"``, which renders as :math:`\alpha`. Default is None, which uses
        ``name``.
    extend_subscript : bool or int, optional
        Move the last underscore-separated tokens of the name into the index subscript, so that ``"P_VA"`` indexed by
        ``i`` renders as :math:`P_{\mathrm{VA}, i}`. True moves one token, and an int moves that many. Default is
        False.
    model : CGEModel, optional
        Model to register the parameter on. Default is None, which uses the model of the enclosing ``with`` block.

    Returns
    -------
    parameter : XTensorVariable
        The parameter as a symbolic xtensor, to build equations from.

    Examples
    --------
    Declare a capital share that renders as a Greek letter:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel() as model:
            alpha = cm.Parameter("alpha", latex_name="alpha", description="Capital share of output")

        model.parameter_names  # ['alpha']
    """
    return modelcontext(model)._register_model_object(
        name=name,
        dims=dims,
        kind="parameter",
        description=description,
        latex_name=latex_name,
        extend_subscript=extend_subscript,
    )


def Equation(name: str, lhs: Any, rhs: Any, *, model: "CGEModel | None" = None) -> XTensorEquation:
    """
    Register the equation ``lhs = rhs`` on a :class:`~cge_modeling.base.cge.CGEModel`.

    The solver drives the residual ``lhs - rhs`` to zero. An equation over indexed objects holds once for each label
    combination of its dims.

    Parameters
    ----------
    name : str
        Name of the equation, unique in the model.
    lhs : XTensorVariable or float
        Left-hand side, built from the model's variables and parameters.
    rhs : XTensorVariable or float
        Right-hand side, built from the model's variables and parameters.
    model : CGEModel, optional
        Model to register the equation on. Default is None, which uses the model of the enclosing ``with`` block.

    Returns
    -------
    equation : XTensorEquation
        The registered equation, holding ``lhs``, ``rhs`` and the residual.

    Examples
    --------
    Make each sector's output its labor productivity times its labor, which registers one equation per sector:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector")
            L = cm.Parameter("L", dims="sector")
            A = cm.Parameter("A", dims="sector")
            cm.Equation("Production", Y, A * L)

        model.unpacked_equation_names  # ['Production [sector=Agriculture]', 'Production [sector=Industry]']
    """
    return modelcontext(model)._register_model_equation(name=name, lhs=lhs, rhs=rhs)
