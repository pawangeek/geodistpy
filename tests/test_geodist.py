import numpy as np
import pytest

# Import the geodist function and any other necessary dependencies
from geodistpy import geodist


# Define test cases using pytest
@pytest.mark.parametrize(
    "coords1, coords2, metric, expected_distance",
    [
        ((52.5200, 13.4050), (48.8566, 2.3522), "km", 879.699316),
        (
            [(37.7749, -122.4194), (34.0522, -118.2437)],
            [(40.7128, -74.0060), (41.8781, -87.6298)],
            "mile",
            [2571.945757, 1745.768063],
        ),
        ((37.7749, -122.4194), (37.7749, -122.4194), "meter", 0.0),
        ((37.7749, -122.4194), (37.7749, -122.4194), "km", 0.0),
        ((0.0, 0.0), (0.0, 0.0), "nmi", 0.0),
        ((90.0, 0.0), (90.0, 0.0), "mile", 0.0),
        ((90.0, 0.0), (-90.0, 0.0), "meter", 20003931.458623),
        ((95.0, 13.4050), (48.8566, -100.0), "km", ValueError),
        ((52.5200, 190.0), (-200.0, 2.3522), "mile", ValueError),
    ],
)
def test_geodist(coords1, coords2, metric, expected_distance):
    if isinstance(expected_distance, type) and issubclass(expected_distance, Exception):
        with pytest.raises(expected_distance):
            geodist(coords1, coords2, metric)
    else:
        distance = geodist(coords1, coords2, metric)
        np.testing.assert_allclose(distance, expected_distance, rtol=1e-6)


# Test cases for metric unit conversion
def test_metric_conversion_meter_to_km():
    distance_meter = geodist((0.0, 0.0), (0.001, 0.001), metric="meter")
    distance_km = geodist((0.0, 0.0), (0.001, 0.001), metric="km")
    assert distance_meter == pytest.approx(distance_km * 1000.0, abs=1e-6)


def test_metric_conversion_mile_to_nmi():
    distance_mile = geodist((0.0, 0.0), (1.0, 1.0), metric="mile")
    distance_nmi = geodist((0.0, 0.0), (1.0, 1.0), metric="nmi")
    assert distance_mile == pytest.approx(distance_nmi * 1.1507795, abs=1e-3)
