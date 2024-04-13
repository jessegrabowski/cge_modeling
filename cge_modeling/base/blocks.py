from abc import ABC
from typing import Union

import numpy as np

from cge_modeling.base.primitives import Equation, Parameter, Variable


class Block(ABC):
    """
    A block is a collection of equations and their associated variables and parameters that can be jointly constructed
    and added to the model.

    A block also holds information about how to do forward and backward calibration of the block.
    """

    __slots__ = [
        "_name",
        "_equations",
        "_variables",
        "_parameters",
        "_forward_inputs",
        "_forward_outputs",
        "_backward_inputs",
        "_backward_outputs",
    ]

    def __init__(self):
        raise NotImplementedError

    def forward_calibration(self) -> dict[str, Union[float, np.array]]:
        """
        Perform forward calibration of the block, mapping parameters to variables.
        """
        raise NotImplementedError

    def backward_calibration(self) -> dict[str, Union[float, np.array]]:
        """
        Perform backward calibration of the block, mapping variables to parameters.
        """
        raise NotImplementedError

    @property
    def equations(self) -> list[Equation]:
        """
        The equations in the block.
        """
        return self._equations

    @property
    def variables(self) -> list[Variable]:
        """
        The variables in the block.
        """
        return self._variables

    @property
    def parameters(self) -> list[Parameter]:
        """
        The parameters in the block.
        """
        return self._parameters

    @property
    def name(self) -> str:
        """
        The name of the block.
        """
        return self._name

    @property
    def forward_inputs(self) -> list[str]:
        """
        The names of the parameters that are inputs to the forward calibration.
        """
        return self._forward_inputs

    @property
    def forward_outputs(self) -> list[str]:
        """
        The names of the variables that are outputs of the forward calibration.
        """
        return self._forward_outputs

    @property
    def backward_inputs(self) -> list[str]:
        """
        The names of the variables that are inputs to the backward calibration.
        """
        return self._backward_inputs

    @property
    def backward_outputs(self) -> list[str]:
        """
        The names of the parameters that are outputs of the backward calibration.
        """
        return self._backward_outputs


class CESProductionBlock(Block):
    def __init__(
        self,
        name: str,
        factors: list[str],
        factor_prices: list[str],
        output: str,
        output_price: str,
        share_parameters: list[str],
        elasticity_parameters: list[str],
        technology_parameters: list[str],
    ) -> None:
        self.name = name
        self.equations = []
        self.variables = []
        self.parameters = []
        self.forward_inputs = []
        self.forward_outputs = []
        self.backward_inputs = []
        self.backward_outputs = []
