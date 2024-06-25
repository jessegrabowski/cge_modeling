import pytest

from cge_modeling.plotting import plot_kateplot, plot_lines
from tests.utilities.models import calibrate_model_2, load_model_2, model_2_data


@pytest.fixture(scope="session")
def test_data():
    mod = load_model_2()
    calibrated_data = calibrate_model_2(**model_2_data)
    idata = mod.simulate(calibrated_data, final_delta_pct={"L_s": 0.5})

    return idata, mod, calibrated_data


def test_plot_kateplot(test_data: tuple):
    plot_data, mod, calibrated_data = test_data
    plot_kateplot(
        plot_data,
        mod=mod,
        initial_values=calibrated_data,
        shock_name="L_s",
        var_names="C",
    )


def test_plot_lines(test_data: tuple):
    plot_data, mod, calibrated_data = test_data
    plot_lines(plot_data, mod=mod)
