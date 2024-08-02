import numpy as np

from scipy import stats

from cge_modeling import Parameter, Variable
from cge_modeling.base.utilities import infer_object_shape_from_coords

DISTRIBUTION_FACTORY = {
    "A": stats.gamma(a=2, scale=1),
    "alpha": stats.beta(a=3, b=3),
    "alpha_VA": stats.beta(a=3, b=3),
    "phi_VA": stats.gamma(a=2, scale=1),
    "psi_VA": stats.gamma(a=2, scale=1),
    "psi_VC": stats.gamma(a=2, scale=1),
    "psi_X": stats.gamma(a=2, scale=1),
    "gamma": stats.beta(a=3, b=3),
    "K_d": stats.truncnorm(loc=4000, scale=100, a=0, b=np.inf),
    "L_d": stats.truncnorm(loc=7000, scale=100, a=0, b=np.inf),
    "K_s": stats.truncnorm(loc=4000, scale=100, a=0, b=np.inf),
    "L_s": stats.truncnorm(loc=7000, scale=100, a=0, b=np.inf),
    "Y": stats.truncnorm(loc=11000, scale=1000, a=0, b=np.inf),
    "C": stats.truncnorm(loc=11000, scale=1000, a=0, b=np.inf),
    "U": stats.norm(loc=0, scale=1000),
    "VA": stats.truncnorm(loc=5000, scale=1000, a=0, b=np.inf),
    "VC": stats.truncnorm(loc=5000, scale=1000, a=0, b=np.inf),
    "X": stats.truncnorm(loc=5000, scale=1000, a=0, b=np.inf),
    "income": stats.truncnorm(loc=11000, scale=1000, a=0, b=np.inf),
    "P": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "P_VA": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "P_VC": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "P_Ag_bar": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "r": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "w": stats.truncnorm(loc=1, scale=0.1, a=0, b=np.inf),
    "resid": stats.norm(loc=0, scale=1),
}


def generate_data(objects: list[Variable | Parameter], coords: dict[str, list[str, ...]]):
    data_dict = {}
    for obj in objects:
        if obj.name in DISTRIBUTION_FACTORY:
            shape = infer_object_shape_from_coords(obj, coords)
            data_dict[obj.name] = DISTRIBUTION_FACTORY[obj.name].rvs(size=shape)
        else:
            raise ValueError(f"No distribution for {obj.name}")
    return data_dict
