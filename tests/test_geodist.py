import numpy as np
import pytest

from geodistpy import geodist, geodist_matrix, greatcircle, greatcircle_matrix
from geodistpy.distance import _get_conv_factor
from geodistpy.geodesic import (
    geodesic_vincenty_inverse,
    geodesic_vincenty,
    great_circle,
    great_circle_array,
    geodist_dimwise,
    geodist_dimwise_harvesine,
)


# ---------------------------------------------------------------------------
# geodist – parametrized core tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "coords1, coords2, metric, expected_distance",
    [
        # Berlin → Paris
        ((52.5200, 13.4050), (48.8566, 2.3522), "km", 879.699316),
        # Multiple points (US cities)
        (
            [(37.7749, -122.4194), (34.0522, -118.2437)],
            [(40.7128, -74.0060), (41.8781, -87.6298)],
            "mile",
            [2571.945757, 1745.768063],
        ),
        # Coincident points – various metrics
        ((37.7749, -122.4194), (37.7749, -122.4194), "meter", 0.0),
        ((37.7749, -122.4194), (37.7749, -122.4194), "km", 0.0),
        ((0.0, 0.0), (0.0, 0.0), "nmi", 0.0),
        ((90.0, 0.0), (90.0, 0.0), "mile", 0.0),
        # Pole to pole
        ((90.0, 0.0), (-90.0, 0.0), "meter", 20003931.458623),
        # Invalid latitude
        ((95.0, 13.4050), (48.8566, -100.0), "km", ValueError),
        # Invalid longitude
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


# ---------------------------------------------------------------------------
# geodist – metric conversion tests
# ---------------------------------------------------------------------------
def test_metric_conversion_meter_to_km():
    distance_meter = geodist((0.0, 0.0), (0.001, 0.001), metric="meter")
    distance_km = geodist((0.0, 0.0), (0.001, 0.001), metric="km")
    assert distance_meter == pytest.approx(distance_km * 1000.0, abs=1e-6)


def test_metric_conversion_mile_to_nmi():
    distance_mile = geodist((0.0, 0.0), (1.0, 1.0), metric="mile")
    distance_nmi = geodist((0.0, 0.0), (1.0, 1.0), metric="nmi")
    assert distance_mile == pytest.approx(distance_nmi * 1.1507795, abs=1e-3)


def test_unsupported_metric():
    with pytest.raises(ValueError, match="not supported"):
        geodist((0.0, 0.0), (1.0, 1.0), metric="furlongs")


# ---------------------------------------------------------------------------
# geodist – symmetry & triangle inequality
# ---------------------------------------------------------------------------
def test_geodist_symmetry():
    """Distance from A→B must equal B→A."""
    a = (52.5200, 13.4050)
    b = (48.8566, 2.3522)
    assert geodist(a, b) == pytest.approx(geodist(b, a), rel=1e-10)


def test_geodist_triangle_inequality():
    """d(A,C) <= d(A,B) + d(B,C)."""
    a = (52.5200, 13.4050)  # Berlin
    b = (48.8566, 2.3522)  # Paris
    c = (40.7128, -74.0060)  # New York
    ab = geodist(a, b)
    bc = geodist(b, c)
    ac = geodist(a, c)
    assert ac <= ab + bc + 1e-6


# ---------------------------------------------------------------------------
# geodist – geographic edge cases
# ---------------------------------------------------------------------------
def test_geodist_equator():
    """Two points on the equator."""
    d = geodist((0.0, 0.0), (0.0, 90.0), metric="km")
    assert 10000 < d < 10100  # ~quarter circumference


def test_geodist_same_longitude():
    """Two points on the same meridian."""
    d = geodist((0.0, 0.0), (90.0, 0.0), metric="km")
    assert 10000 < d < 10100


def test_geodist_antimeridian():
    """Points straddling the antimeridian (date line)."""
    d = geodist((0.0, 179.0), (0.0, -179.0), metric="km")
    assert d == pytest.approx(222.4, rel=0.01)


def test_geodist_antipodal():
    """Nearly-antipodal points (Vincenty convergence stress test)."""
    d = geodist((0.0, 0.0), (0.5, 179.5), metric="km")
    assert d > 19000


# ---------------------------------------------------------------------------
# geodist_matrix – pdist mode (single list)
# ---------------------------------------------------------------------------
def test_geodist_matrix_pdist():
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (40.7128, -74.0060)]
    mat = geodist_matrix(coords, metric="km")
    assert mat.shape == (3, 3)
    # Diagonal should be zero
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
    # Symmetric
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)
    # Berlin–Paris should match geodist
    d_bp = geodist((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    assert mat[0, 1] == pytest.approx(d_bp, rel=1e-6)


# ---------------------------------------------------------------------------
# geodist_matrix – cdist mode (two lists)
# ---------------------------------------------------------------------------
def test_geodist_matrix_cdist():
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = geodist_matrix(coords1, coords2, metric="mile")
    assert mat.shape == (2, 2)
    # Each entry should match individual geodist calls
    for i in range(2):
        for j in range(2):
            expected = geodist(coords1[i], coords2[j], metric="mile")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_geodist_matrix_invalid_coords():
    with pytest.raises(ValueError):
        geodist_matrix([(95.0, 0.0), (0.0, 0.0)])


# ---------------------------------------------------------------------------
# greatcircle – single pair
# ---------------------------------------------------------------------------
def test_greatcircle_single_pair():
    d = greatcircle((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    # Great circle should be close to Vincenty (within ~0.5% for moderate distances)
    d_vincenty = geodist((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    assert d == pytest.approx(d_vincenty, rel=0.005)


def test_greatcircle_coincident():
    d = greatcircle((37.7749, -122.4194), (37.7749, -122.4194))
    assert d == pytest.approx(0.0, abs=1e-6)


def test_greatcircle_multiple_pairs():
    coords1 = [(37.7749, -122.4194), (34.0522, -118.2437)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    d = greatcircle(coords1, coords2, metric="km")
    assert d.shape == (2,)
    assert all(d > 0)


def test_greatcircle_invalid_latitude():
    with pytest.raises(ValueError):
        greatcircle(
            [(95.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0)],
        )


def test_greatcircle_invalid_longitude():
    with pytest.raises(ValueError):
        greatcircle(
            [(0.0, 200.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0)],
        )


# ---------------------------------------------------------------------------
# greatcircle_matrix – pdist mode
# ---------------------------------------------------------------------------
def test_greatcircle_matrix_pdist():
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
    mat = greatcircle_matrix(coords, metric="km")
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)


# ---------------------------------------------------------------------------
# greatcircle_matrix – cdist mode
# ---------------------------------------------------------------------------
def test_greatcircle_matrix_cdist():
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = greatcircle_matrix(coords1, coords2, metric="km")
    assert mat.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            expected = greatcircle(coords1[i], coords2[j], metric="km")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_greatcircle_matrix_invalid_coords():
    with pytest.raises(ValueError):
        greatcircle_matrix([(0.0, 200.0), (0.0, 0.0)])


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty_inverse
# ---------------------------------------------------------------------------
def test_vincenty_inverse_coincident():
    assert geodesic_vincenty_inverse((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_vincenty_inverse_known_distance():
    """Berlin–Paris is ~879 km."""
    d = geodesic_vincenty_inverse((52.5200, 13.4050), (48.8566, 2.3522))
    assert d == pytest.approx(879699.316, rel=1e-3)


def test_vincenty_inverse_pole_to_pole():
    d = geodesic_vincenty_inverse((90.0, 0.0), (-90.0, 0.0))
    assert d == pytest.approx(20003931.458623, rel=1e-6)


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty (with fallback)
# ---------------------------------------------------------------------------
def test_geodesic_vincenty_normal():
    d = geodesic_vincenty((52.5200, 13.4050), (48.8566, 2.3522))
    assert d == pytest.approx(879699.316, rel=1e-3)


def test_geodesic_vincenty_coincident():
    assert geodesic_vincenty((10.0, 20.0), (10.0, 20.0)) == 0.0


# ---------------------------------------------------------------------------
# Low-level: great_circle (scalar)
# ---------------------------------------------------------------------------
def test_great_circle_scalar():
    d = great_circle((52.5200, 13.4050), (48.8566, 2.3522))
    assert d > 870_000 and d < 890_000  # ~879 km


def test_great_circle_coincident():
    assert great_circle((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0, abs=1e-8)


def test_great_circle_quarter_circumference():
    """Equator to north pole is ~10,000 km."""
    d = great_circle((0.0, 0.0), (90.0, 0.0))
    assert d == pytest.approx(6371009 * np.pi / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# Low-level: great_circle_array (numpy-based)
# ---------------------------------------------------------------------------
def test_great_circle_array_single():
    d = great_circle_array(np.array([52.5200, 13.4050]), np.array([48.8566, 2.3522]))
    assert d > 870_000 and d < 890_000


def test_great_circle_array_multiple():
    u = np.array([[52.5200, 13.4050], [37.7749, -122.4194]])
    v = np.array([[48.8566, 2.3522], [40.7128, -74.0060]])
    d = great_circle_array(u.T, v.T)
    # Should be an array-like result
    assert d.shape == (2,) or len(d) == 2


# ---------------------------------------------------------------------------
# geodist_dimwise
# ---------------------------------------------------------------------------
def test_geodist_dimwise_shape():
    X = np.array(
        [
            [52.5200, 13.4050, 100],
            [48.8566, 2.3522, 200],
            [40.7128, -74.0060, 300],
        ]
    )
    dist = geodist_dimwise(X)
    assert dist.shape == (3, 3, 2)  # n_features - 1 = 2


def test_geodist_dimwise_diagonal_zero():
    X = np.array(
        [
            [52.5200, 13.4050, 100],
            [48.8566, 2.3522, 200],
        ]
    )
    dist = geodist_dimwise(X)
    # Geodesic dimension: diagonal is 0
    np.testing.assert_allclose(np.diag(dist[:, :, 0]), 0.0, atol=1e-4)
    # Extra dimension: diagonal is 0
    np.testing.assert_allclose(np.diag(dist[:, :, 1]), 0.0, atol=1e-8)


def test_geodist_dimwise_extra_dim_differences():
    X = np.array(
        [
            [0.0, 0.0, 10],
            [0.0, 0.0, 30],
        ]
    )
    dist = geodist_dimwise(X)
    # Geodesic dim should be 0 (same lat/lon)
    assert dist[0, 1, 0] == pytest.approx(0.0, abs=1e-4)
    # Extra dim should be the raw difference (X[i] - X[j])
    assert dist[0, 1, 1] == pytest.approx(-20.0, abs=1e-8)
    assert dist[1, 0, 1] == pytest.approx(20.0, abs=1e-8)


# ---------------------------------------------------------------------------
# geodist_dimwise_harvesine
# ---------------------------------------------------------------------------
def test_geodist_dimwise_haversine_shape():
    X = np.array(
        [
            [52.5200, 13.4050, 100],
            [48.8566, 2.3522, 200],
            [40.7128, -74.0060, 300],
        ]
    )
    sdist = geodist_dimwise_harvesine(X)
    assert sdist.shape == (3, 3, 3)  # same as n_features


def test_geodist_dimwise_haversine_diagonal_zero():
    X = np.array(
        [
            [52.5200, 13.4050, 100],
            [48.8566, 2.3522, 200],
        ]
    )
    sdist = geodist_dimwise_harvesine(X)
    # All diagonals should be zero (squared distances)
    for k in range(3):
        np.testing.assert_allclose(np.diag(sdist[:, :, k]), 0.0, atol=1e-4)


def test_geodist_dimwise_haversine_nonnegative():
    """Squared distances should be non-negative."""
    X = np.array(
        [
            [52.5200, 13.4050, 100],
            [48.8566, 2.3522, 200],
            [40.7128, -74.0060, 300],
        ]
    )
    sdist = geodist_dimwise_harvesine(X)
    assert np.all(sdist >= -1e-10)


# ---------------------------------------------------------------------------
# _get_conv_factor
# ---------------------------------------------------------------------------
def test_conv_factor_values():
    assert _get_conv_factor("meter") == 1
    assert _get_conv_factor("km") == 1e-3
    assert _get_conv_factor("mile") == pytest.approx(1 / 1609.344)
    assert _get_conv_factor("nmi") == pytest.approx(1 / 1852.0)


def test_conv_factor_invalid():
    with pytest.raises(ValueError):
        _get_conv_factor("invalid_metric")


# ---------------------------------------------------------------------------
# Vincenty vs Great Circle comparison
# ---------------------------------------------------------------------------
def test_vincenty_vs_greatcircle_close_for_short_distances():
    """For short distances the two methods should agree within ~0.3%."""
    p1 = (48.8566, 2.3522)  # Paris
    p2 = (48.8606, 2.3376)  # ~500m away
    d_vincenty = geodist(p1, p2, metric="meter")
    d_gc = greatcircle(p1, p2, metric="meter")
    assert d_vincenty == pytest.approx(d_gc, rel=0.003)


def test_vincenty_vs_greatcircle_diverge_for_long_distances():
    """For very long distances, Vincenty (ellipsoid) and Great Circle (sphere) may differ."""
    p1 = (0.0, 0.0)
    p2 = (0.0, 180.0)  # half the equator
    d_vincenty = geodist(p1, p2, metric="km")
    d_gc = greatcircle(p1, p2, metric="km")
    # Both should be roughly 20,000 km but differ because sphere vs ellipsoid
    assert abs(d_vincenty - d_gc) > 0  # they should not be identical
    assert d_vincenty > 19000
    assert d_gc > 19000


# ---------------------------------------------------------------------------
# Consistency: matrix entries match pairwise calls
# ---------------------------------------------------------------------------
def test_geodist_matrix_matches_pairwise():
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (40.7128, -74.0060)]
    mat = geodist_matrix(coords, metric="meter")
    for i in range(len(coords)):
        for j in range(len(coords)):
            expected = geodist(coords[i], coords[j], metric="meter")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)
