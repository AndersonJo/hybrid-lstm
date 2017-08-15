from tools import load_household_power_consumption, to_timeseries


def test_load_dataset():
    dataset, data_x, data_y = load_household_power_consumption()
    assert data_x.shape == (43642, 7)
    assert data_y.shape == (43642, 1)
    assert to_timeseries(data_x).shape == (43294, 30, 7)
    assert to_timeseries(data_y).shape == (43613, 30, 1)
