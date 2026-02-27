"""Tests for geodistpy distance calculation functions."""

import numpy as np
import pytest

from geodistpy import (
    geodist,
    geodist_matrix,
    greatcircle,
    greatcircle_matrix,
    bearing,
    destination,
    interpolate,
    midpoint,
    point_in_radius,
    geodesic_knn,
)
from geodistpy.distance import _get_conv_factor
from geodistpy.geodesic import (
    geodesic_vincenty_inverse,
    geodesic_vincenty,
    geodesic_vincenty_inverse_full,
    geodesic_vincenty_direct,
    great_circle,
    great_circle_array,
    geodist_dimwise,
    geodist_dimwise_harvesine,
    ELLIPSOIDS,
    _resolve_ellipsoid,
)


# ---------------------------------------------------------------------------
# geodist - parametrized core tests
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
        # Coincident points - various metrics
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
    """Verify geodist returns correct distances or raises on invalid input."""
    if isinstance(expected_distance, type) and issubclass(expected_distance, Exception):
        with pytest.raises(expected_distance):
            geodist(coords1, coords2, metric)
    else:
        distance = geodist(coords1, coords2, metric)
        np.testing.assert_allclose(distance, expected_distance, rtol=1e-6)


# ---------------------------------------------------------------------------
# geodist - metric conversion tests
# ---------------------------------------------------------------------------
def test_metric_conversion_meter_to_km():
    """Verify meter and km results are consistent (factor of 1000)."""
    distance_meter = geodist((0.0, 0.0), (0.001, 0.001), metric="meter")
    distance_km = geodist((0.0, 0.0), (0.001, 0.001), metric="km")
    assert distance_meter == pytest.approx(distance_km * 1000.0, abs=1e-6)


def test_metric_conversion_mile_to_nmi():
    """Verify mile and nautical mile results are consistent."""
    distance_mile = geodist((0.0, 0.0), (1.0, 1.0), metric="mile")
    distance_nmi = geodist((0.0, 0.0), (1.0, 1.0), metric="nmi")
    assert distance_mile == pytest.approx(distance_nmi * 1.1507795, abs=1e-3)


def test_unsupported_metric():
    """Verify ValueError is raised for an unsupported metric."""
    with pytest.raises(ValueError, match="not supported"):
        geodist((0.0, 0.0), (1.0, 1.0), metric="furlongs")


# ---------------------------------------------------------------------------
# geodist - symmetry & triangle inequality
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
# geodist - geographic edge cases
# ---------------------------------------------------------------------------
def test_geodist_equator():
    """Two points on the equator separated by 90 degrees longitude."""
    d = geodist((0.0, 0.0), (0.0, 90.0), metric="km")
    assert 10000 < d < 10100  # ~quarter circumference


def test_geodist_same_longitude():
    """Two points on the same meridian from equator to pole."""
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
# geodist_matrix - pdist mode (single list)
# ---------------------------------------------------------------------------
def test_geodist_matrix_pdist():
    """Verify pdist-mode matrix is symmetric with zero diagonal."""
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (40.7128, -74.0060)]
    mat = geodist_matrix(coords, metric="km")
    assert mat.shape == (3, 3)
    # Diagonal should be zero
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
    # Symmetric
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)
    # Berlin-Paris should match geodist
    d_bp = geodist((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    assert mat[0, 1] == pytest.approx(d_bp, rel=1e-6)


# ---------------------------------------------------------------------------
# geodist_matrix - cdist mode (two lists)
# ---------------------------------------------------------------------------
def test_geodist_matrix_cdist():
    """Verify cdist-mode matrix entries match individual geodist calls."""
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = geodist_matrix(coords1, coords2, metric="mile")
    assert mat.shape == (2, 2)
    # Each entry should match individual geodist calls
    for i in range(2):
        for j in range(2):
            expected = geodist(coords1[i], coords2[j], metric="mile")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_geodist_matrix_cdist_different_sizes():
    """Verify cdist-mode works when coords1 and coords2 have different lengths."""
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = geodist_matrix(coords1, coords2, metric="km")
    assert mat.shape == (3, 2)
    # Each entry should match individual geodist calls
    for i in range(3):
        for j in range(2):
            expected = geodist(coords1[i], coords2[j], metric="km")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_greatcircle_matrix_cdist_different_sizes():
    """Verify cdist-mode greatcircle_matrix works with different-length inputs."""
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = greatcircle_matrix(coords1, coords2, metric="km")
    assert mat.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            expected = greatcircle(coords1[i], coords2[j], metric="km")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_geodist_matrix_invalid_coords():
    """Verify ValueError is raised for out-of-range coordinates."""
    with pytest.raises(ValueError):
        geodist_matrix([(95.0, 0.0), (0.0, 0.0)])


def test_geodist_multi_invalid_latitude():
    """Verify ValueError for out-of-range latitude in multi-point geodist."""
    with pytest.raises(ValueError):
        geodist([(95.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)])


def test_geodist_multi_invalid_longitude():
    """Verify ValueError for out-of-range longitude in multi-point geodist."""
    with pytest.raises(ValueError):
        geodist([(0.0, 200.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)])


def test_geodist_multi_wrong_shape():
    """Verify ValueError when coords have wrong number of columns."""
    with pytest.raises((ValueError, AssertionError)):
        geodist([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])


def test_geodist_matrix_cdist_invalid_coords2():
    """Verify ValueError for invalid coords2 in cdist mode."""
    with pytest.raises(ValueError):
        geodist_matrix([(0.0, 0.0), (1.0, 1.0)], [(95.0, 0.0), (0.0, 0.0)])


def test_geodist_single_point_wrong_shape():
    """Verify ValueError for single point with wrong array shape."""
    with pytest.raises(ValueError):
        geodist([[52.5, 13.4]], [[48.8, 2.3]])


def test_geodist_single_point_invalid_longitude_only():
    """Verify ValueError for single point with valid lat but invalid lon."""
    with pytest.raises(ValueError):
        geodist((52.5, 190.0), (48.8, 2.3))


def test_geodist_matrix_wrong_shape():
    """Verify ValueError for matrix input with wrong number of columns."""
    with pytest.raises((ValueError, IndexError)):
        geodist_matrix([(0.0,), (1.0,)])


def test_greatcircle_wrong_shape():
    """Verify ValueError for greatcircle with wrong number of columns."""
    with pytest.raises(ValueError):
        greatcircle(
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        )


def test_greatcircle_matrix_wrong_shape():
    """Verify ValueError for greatcircle_matrix with wrong columns."""
    with pytest.raises((ValueError, IndexError)):
        greatcircle_matrix([(0.0,), (1.0,)])


# ---------------------------------------------------------------------------
# greatcircle - single pair
# ---------------------------------------------------------------------------
def test_greatcircle_single_pair():
    """Verify great circle result is close to Vincenty for moderate distances."""
    d = greatcircle((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    d_vincenty = geodist((52.5200, 13.4050), (48.8566, 2.3522), metric="km")
    assert d == pytest.approx(d_vincenty, rel=0.005)


def test_greatcircle_coincident():
    """Verify zero distance for coincident points."""
    d = greatcircle((37.7749, -122.4194), (37.7749, -122.4194))
    assert d == pytest.approx(0.0, abs=1e-6)


def test_greatcircle_multiple_pairs():
    """Verify great circle handles multiple coordinate pairs."""
    coords1 = [(37.7749, -122.4194), (34.0522, -118.2437)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    d = greatcircle(coords1, coords2, metric="km")
    assert d.shape == (2,)
    assert all(d > 0)


def test_greatcircle_invalid_latitude():
    """Verify ValueError for latitude out of range."""
    with pytest.raises(ValueError):
        greatcircle(
            [(95.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0)],
        )


def test_greatcircle_invalid_longitude():
    """Verify ValueError for longitude out of range."""
    with pytest.raises(ValueError):
        greatcircle(
            [(0.0, 200.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0)],
        )


# ---------------------------------------------------------------------------
# greatcircle_matrix - pdist mode
# ---------------------------------------------------------------------------
def test_greatcircle_matrix_pdist():
    """Verify pdist-mode great circle matrix is symmetric with zero diagonal."""
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
    mat = greatcircle_matrix(coords, metric="km")
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)


# ---------------------------------------------------------------------------
# greatcircle_matrix - cdist mode
# ---------------------------------------------------------------------------
def test_greatcircle_matrix_cdist():
    """Verify cdist-mode great circle matrix entries match pairwise calls."""
    coords1 = [(52.5200, 13.4050), (48.8566, 2.3522)]
    coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
    mat = greatcircle_matrix(coords1, coords2, metric="km")
    assert mat.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            expected = greatcircle(coords1[i], coords2[j], metric="km")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


def test_greatcircle_matrix_invalid_coords():
    """Verify ValueError for invalid coordinates in greatcircle_matrix."""
    with pytest.raises(ValueError):
        greatcircle_matrix([(0.0, 200.0), (0.0, 0.0)])


def test_greatcircle_matrix_cdist_invalid_coords2():
    """Verify ValueError for invalid coords2 in greatcircle_matrix cdist mode."""
    with pytest.raises(ValueError):
        greatcircle_matrix([(0.0, 0.0), (1.0, 1.0)], [(95.0, 0.0), (0.0, 0.0)])


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty_inverse
# ---------------------------------------------------------------------------
def test_vincenty_inverse_coincident():
    """Verify zero distance for coincident points via Vincenty inverse."""
    assert geodesic_vincenty_inverse((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_vincenty_inverse_known_distance():
    """Berlin to Paris is approximately 879 km."""
    d = geodesic_vincenty_inverse((52.5200, 13.4050), (48.8566, 2.3522))
    assert d == pytest.approx(879699.316, rel=1e-3)


def test_vincenty_inverse_pole_to_pole():
    """Verify pole-to-pole distance via Vincenty inverse."""
    d = geodesic_vincenty_inverse((90.0, 0.0), (-90.0, 0.0))
    assert d == pytest.approx(20003931.458623, rel=1e-6)


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty (with fallback)
# ---------------------------------------------------------------------------
def test_geodesic_vincenty_normal():
    """Verify Vincenty with fallback returns correct Berlin-Paris distance."""
    d = geodesic_vincenty((52.5200, 13.4050), (48.8566, 2.3522))
    assert d == pytest.approx(879699.316, rel=1e-3)


def test_geodesic_vincenty_coincident():
    """Verify zero distance for coincident points via Vincenty with fallback."""
    assert geodesic_vincenty((10.0, 20.0), (10.0, 20.0)) == 0.0


# ---------------------------------------------------------------------------
# Low-level: great_circle (scalar)
# ---------------------------------------------------------------------------
def test_great_circle_scalar():
    """Verify scalar great circle distance for Berlin-Paris."""
    d = great_circle((52.5200, 13.4050), (48.8566, 2.3522))
    assert d > 870_000 and d < 890_000  # ~879 km


def test_great_circle_coincident():
    """Verify zero distance for coincident points via great circle."""
    assert great_circle((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0, abs=1e-8)


def test_great_circle_quarter_circumference():
    """Equator to north pole is approximately 10,002 km (WGS84 geodesic)."""
    d = great_circle((0.0, 0.0), (90.0, 0.0))
    # True WGS84 geodesic distance equator-to-pole is ~10,001,965.729 m.
    # Andoyer-Lambert correction gives ~10,001,958.7 m (within ~7 m).
    assert d == pytest.approx(10_001_965.729, rel=1e-3)


# ---------------------------------------------------------------------------
# Low-level: great_circle_array (numpy-based)
# ---------------------------------------------------------------------------
def test_great_circle_array_single():
    """Verify great_circle_array with a single coordinate pair."""
    d = great_circle_array(np.array([52.5200, 13.4050]), np.array([48.8566, 2.3522]))
    assert d > 870_000 and d < 890_000


def test_great_circle_array_multiple():
    """Verify great_circle_array with multiple coordinate pairs."""
    u = np.array([[52.5200, 13.4050], [37.7749, -122.4194]])
    v = np.array([[48.8566, 2.3522], [40.7128, -74.0060]])
    d = great_circle_array(u.T, v.T)
    # Should be an array-like result
    assert d.shape == (2,) or len(d) == 2


# ---------------------------------------------------------------------------
# geodist_dimwise
# ---------------------------------------------------------------------------
def test_geodist_dimwise_shape():
    """Verify output shape of geodist_dimwise is (n, n, n_features - 1)."""
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
    """Verify diagonal entries are zero in geodist_dimwise output."""
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
    """Verify extra dimensions contain raw signed differences."""
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
    """Verify output shape of geodist_dimwise_harvesine is (n, n, n_features)."""
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
    """Verify diagonal entries are zero in haversine squared distances."""
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
    """Verify conversion factors for all supported metrics."""
    assert _get_conv_factor("meter") == 1
    assert _get_conv_factor("km") == 1e-3
    assert _get_conv_factor("mile") == pytest.approx(1 / 1609.344)
    assert _get_conv_factor("nmi") == pytest.approx(1 / 1852.0)


def test_conv_factor_invalid():
    """Verify ValueError for an unsupported metric in _get_conv_factor."""
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
    """For very long distances, Vincenty and Great Circle should differ."""
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
    """Verify every matrix entry matches the corresponding pairwise geodist call."""
    coords = [(52.5200, 13.4050), (48.8566, 2.3522), (40.7128, -74.0060)]
    mat = geodist_matrix(coords, metric="meter")
    for i, c1 in enumerate(coords):
        for j, c2 in enumerate(coords):
            expected = geodist(c1, c2, metric="meter")
            assert mat[i, j] == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# NEW FEATURES — bearing, destination, interpolate/midpoint,
#                point_in_radius, geodesic_knn
# ===========================================================================


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty_inverse_full
# ---------------------------------------------------------------------------
def test_vincenty_inverse_full_distance_matches():
    """Distance from the full variant must match the distance-only variant."""
    p1 = (52.5200, 13.4050)
    p2 = (48.8566, 2.3522)
    d_only = geodesic_vincenty_inverse(p1, p2)
    d_full, fwd, back = geodesic_vincenty_inverse_full(p1, p2)
    assert d_full == pytest.approx(d_only, rel=1e-10)


def test_vincenty_inverse_full_coincident():
    """Coincident points → distance=0, azimuths=0."""
    d, fwd, back = geodesic_vincenty_inverse_full((10.0, 20.0), (10.0, 20.0))
    assert d == 0.0
    assert fwd == 0.0
    assert back == 0.0


def test_vincenty_inverse_full_due_east():
    """Moving due east on the equator → forward azimuth ~90°."""
    d, fwd, back = geodesic_vincenty_inverse_full((0.0, 0.0), (0.0, 1.0))
    assert fwd == pytest.approx(90.0, abs=0.01)


def test_vincenty_inverse_full_due_north():
    """Moving due north → forward azimuth ~0°."""
    d, fwd, back = geodesic_vincenty_inverse_full((0.0, 0.0), (1.0, 0.0))
    assert fwd == pytest.approx(0.0, abs=0.01)


def test_vincenty_inverse_full_azimuth_range():
    """Azimuths must be in [0, 360)."""
    _, fwd, back = geodesic_vincenty_inverse_full((52.5200, 13.4050), (48.8566, 2.3522))
    assert 0.0 <= fwd < 360.0
    assert 0.0 <= back < 360.0


# ---------------------------------------------------------------------------
# Low-level: geodesic_vincenty_direct
# ---------------------------------------------------------------------------
def test_vincenty_direct_roundtrip():
    """Inverse → Direct roundtrip: start + bearing + distance → end point."""
    p1 = (52.5200, 13.4050)
    p2 = (48.8566, 2.3522)
    d, fwd, _ = geodesic_vincenty_inverse_full(p1, p2)
    lat, lon = geodesic_vincenty_direct(p1, fwd, d)
    assert lat == pytest.approx(p2[0], abs=1e-5)
    assert lon == pytest.approx(p2[1], abs=1e-5)


def test_vincenty_direct_zero_distance():
    """Zero distance should return the starting point."""
    lat, lon = geodesic_vincenty_direct((52.5200, 13.4050), 90.0, 0.0)
    assert lat == pytest.approx(52.5200, abs=1e-8)
    assert lon == pytest.approx(13.4050, abs=1e-8)


def test_vincenty_direct_due_east_equator():
    """111.32 km east on equator ≈ 1° longitude shift."""
    lat, lon = geodesic_vincenty_direct((0.0, 0.0), 90.0, 111_320.0)
    assert lat == pytest.approx(0.0, abs=0.01)
    assert lon == pytest.approx(1.0, abs=0.01)


def test_vincenty_direct_due_north():
    """~111 km north from equator ≈ 1° latitude shift."""
    lat, lon = geodesic_vincenty_direct((0.0, 0.0), 0.0, 110_574.0)
    assert lat == pytest.approx(1.0, abs=0.01)
    assert lon == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# bearing
# ---------------------------------------------------------------------------
def test_bearing_due_east():
    """Due east on the equator → bearing ~90°."""
    b = bearing((0.0, 0.0), (0.0, 1.0))
    assert b == pytest.approx(90.0, abs=0.01)


def test_bearing_due_north():
    """Due north → bearing ~0°."""
    b = bearing((0.0, 0.0), (1.0, 0.0))
    assert b == pytest.approx(0.0, abs=0.01)


def test_bearing_due_south():
    """Due south → bearing ~180°."""
    b = bearing((1.0, 0.0), (0.0, 0.0))
    assert b == pytest.approx(180.0, abs=0.01)


def test_bearing_due_west():
    """Due west on the equator → bearing ~270°."""
    b = bearing((0.0, 1.0), (0.0, 0.0))
    assert b == pytest.approx(270.0, abs=0.01)


def test_bearing_berlin_paris():
    """Berlin → Paris bearing should be roughly southwest (~245°)."""
    b = bearing((52.5200, 13.4050), (48.8566, 2.3522))
    assert 240.0 < b < 250.0


def test_bearing_coincident():
    """Coincident points → bearing = 0."""
    b = bearing((52.5200, 13.4050), (52.5200, 13.4050))
    assert b == pytest.approx(0.0, abs=1e-8)


def test_bearing_range():
    """Bearing must be in [0, 360)."""
    b = bearing((52.5200, 13.4050), (48.8566, 2.3522))
    assert 0.0 <= b < 360.0


def test_bearing_invalid_latitude():
    """ValueError for latitude out of range."""
    with pytest.raises(ValueError):
        bearing((95.0, 0.0), (0.0, 0.0))


def test_bearing_invalid_longitude():
    """ValueError for longitude out of range."""
    with pytest.raises(ValueError):
        bearing((0.0, 200.0), (0.0, 0.0))


def test_bearing_symmetry_not_equal():
    """A→B bearing ≠ B→A bearing in general (they differ by ~180° mod 360)."""
    b_ab = bearing((52.5200, 13.4050), (48.8566, 2.3522))
    b_ba = bearing((48.8566, 2.3522), (52.5200, 13.4050))
    # They should not be equal
    assert abs(b_ab - b_ba) > 1.0
    # But they should be roughly 180° apart (on a great circle, not exact)
    diff = abs(b_ab - b_ba)
    assert 170.0 < diff < 190.0 or 170.0 < (360.0 - diff) < 190.0


# ---------------------------------------------------------------------------
# destination
# ---------------------------------------------------------------------------
def test_destination_roundtrip():
    """destination(A, bearing(A,B), dist(A,B)) ≈ B."""
    a = (52.5200, 13.4050)
    b = (48.8566, 2.3522)
    b_deg = bearing(a, b)
    d = geodist(a, b, metric="km")
    lat, lon = destination(a, b_deg, d, metric="km")
    assert lat == pytest.approx(b[0], abs=1e-4)
    assert lon == pytest.approx(b[1], abs=1e-4)


def test_destination_zero_distance():
    """Zero distance returns the starting point."""
    lat, lon = destination((52.5200, 13.4050), 90.0, 0.0)
    assert lat == pytest.approx(52.5200, abs=1e-8)
    assert lon == pytest.approx(13.4050, abs=1e-8)


def test_destination_due_east_km():
    """~111 km east on the equator ≈ 1° longitude."""
    lat, lon = destination((0.0, 0.0), 90.0, 111.32, metric="km")
    assert lat == pytest.approx(0.0, abs=0.01)
    assert lon == pytest.approx(1.0, abs=0.01)


def test_destination_metric_mile():
    """Verify mile metric works."""
    lat1, lon1 = destination((0.0, 0.0), 0.0, 100, metric="mile")
    lat2, lon2 = destination((0.0, 0.0), 0.0, 100 * 1.609344, metric="km")
    assert lat1 == pytest.approx(lat2, abs=1e-5)
    assert lon1 == pytest.approx(lon2, abs=1e-5)


def test_destination_invalid_latitude():
    """ValueError for invalid latitude."""
    with pytest.raises(ValueError):
        destination((95.0, 0.0), 90.0, 100.0)


def test_destination_invalid_longitude():
    """ValueError for invalid longitude."""
    with pytest.raises(ValueError):
        destination((0.0, 200.0), 90.0, 100.0)


def test_destination_longitude_normalisation():
    """Destination longitude should be normalised to [-180, 180]."""
    # Travel far east past the antimeridian
    _, lon = destination((0.0, 170.0), 90.0, 2000, metric="km")
    assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# interpolate & midpoint
# ---------------------------------------------------------------------------
def test_midpoint_equator():
    """Midpoint of two equatorial points is at the midpoint longitude."""
    mid = midpoint((0.0, 0.0), (0.0, 10.0))
    assert mid[0] == pytest.approx(0.0, abs=0.01)
    assert mid[1] == pytest.approx(5.0, abs=0.05)


def test_midpoint_same_point():
    """Midpoint of identical points is the point itself."""
    mid = midpoint((52.5200, 13.4050), (52.5200, 13.4050))
    assert mid[0] == pytest.approx(52.5200, abs=1e-8)
    assert mid[1] == pytest.approx(13.4050, abs=1e-8)


def test_midpoint_symmetric():
    """Midpoint(A,B) ≈ Midpoint(B,A)."""
    a = (52.5200, 13.4050)
    b = (48.8566, 2.3522)
    m1 = midpoint(a, b)
    m2 = midpoint(b, a)
    assert m1[0] == pytest.approx(m2[0], abs=1e-5)
    assert m1[1] == pytest.approx(m2[1], abs=1e-5)


def test_midpoint_equidistant():
    """Midpoint should be equidistant from both endpoints."""
    a = (52.5200, 13.4050)
    b = (48.8566, 2.3522)
    m = midpoint(a, b)
    d_am = geodist(a, m, metric="meter")
    d_mb = geodist(m, b, metric="meter")
    assert d_am == pytest.approx(d_mb, rel=1e-3)


def test_interpolate_single_is_midpoint():
    """interpolate with n_points=1 must equal midpoint."""
    a = (52.5200, 13.4050)
    b = (48.8566, 2.3522)
    pts = interpolate(a, b, n_points=1)
    m = midpoint(a, b)
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(m[0], abs=1e-8)
    assert pts[0][1] == pytest.approx(m[1], abs=1e-8)


def test_interpolate_multiple_count():
    """interpolate(n_points=4) returns exactly 4 waypoints."""
    pts = interpolate((0.0, 0.0), (0.0, 10.0), n_points=4)
    assert len(pts) == 4


def test_interpolate_multiple_equispaced():
    """Waypoints should be roughly equispaced along the geodesic."""
    a = (0.0, 0.0)
    b = (0.0, 10.0)
    pts = interpolate(a, b, n_points=4)
    # Each segment should be ~1/5 of total distance
    total = geodist(a, b, metric="km")
    all_pts = [a] + pts + [b]
    for i in range(len(all_pts) - 1):
        seg = geodist(all_pts[i], all_pts[i + 1], metric="km")
        assert seg == pytest.approx(total / 5.0, rel=0.01)


def test_interpolate_ordered():
    """Waypoints should be ordered from point1 towards point2."""
    a = (0.0, 0.0)
    b = (0.0, 10.0)
    pts = interpolate(a, b, n_points=3)
    lons = [p[1] for p in pts]
    assert lons == sorted(lons)


def test_interpolate_invalid_n_points():
    """ValueError for n_points < 1."""
    with pytest.raises(ValueError):
        interpolate((0.0, 0.0), (0.0, 10.0), n_points=0)


def test_interpolate_invalid_coords():
    """ValueError for out-of-range coordinates."""
    with pytest.raises(ValueError):
        interpolate((95.0, 0.0), (0.0, 0.0))


# ---------------------------------------------------------------------------
# point_in_radius
# ---------------------------------------------------------------------------
def test_point_in_radius_basic():
    """Paris and London are within 1000 km of Berlin; New York is not."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = point_in_radius((52.5200, 13.4050), pts, 1000, metric="km")
    assert 0 in idx  # Paris ~880 km
    assert 2 in idx  # London ~930 km
    assert 1 not in idx  # New York ~6400 km


def test_point_in_radius_none_inside():
    """No points within a very small radius."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006)]
    idx, dists = point_in_radius((52.5200, 13.4050), pts, 1.0, metric="km")
    assert len(idx) == 0
    assert len(dists) == 0


def test_point_in_radius_all_inside():
    """All points within a very large radius."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = point_in_radius((52.5200, 13.4050), pts, 100000, metric="km")
    assert len(idx) == 3


def test_point_in_radius_distances_correct():
    """Returned distances should match individual geodist calls."""
    center = (52.5200, 13.4050)
    pts = [(48.8566, 2.3522), (51.5074, -0.1278)]
    idx, dists = point_in_radius(center, pts, 1000, metric="km")
    for i_pos, i_orig in enumerate(idx):
        expected = geodist(center, pts[i_orig], metric="km")
        assert dists[i_pos] == pytest.approx(expected, rel=1e-6)


def test_point_in_radius_boundary():
    """A point exactly on the radius boundary should be included (<=)."""
    center = (0.0, 0.0)
    pts = [(0.0, 1.0)]
    d = geodist(center, pts[0], metric="km")
    idx, _ = point_in_radius(center, pts, d, metric="km")
    assert 0 in idx


def test_point_in_radius_invalid_center():
    """ValueError for invalid center coordinates."""
    with pytest.raises(ValueError):
        point_in_radius((95.0, 0.0), [(0.0, 0.0)], 100)


def test_point_in_radius_wrong_shape():
    """ValueError for candidates with wrong shape."""
    with pytest.raises(ValueError):
        point_in_radius((0.0, 0.0), [(0.0,)], 100)


def test_point_in_radius_metric():
    """Verify metric parameter works for miles."""
    center = (52.5200, 13.4050)
    pts = [(48.8566, 2.3522)]
    idx_km, d_km = point_in_radius(center, pts, 900, metric="km")
    idx_mi, d_mi = point_in_radius(center, pts, 900 / 1.609344, metric="mile")
    assert len(idx_km) == len(idx_mi)


# ---------------------------------------------------------------------------
# geodesic_knn
# ---------------------------------------------------------------------------
def test_geodesic_knn_basic():
    """k=2 nearest to Berlin: Paris and London (not New York)."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=2, metric="km")
    assert len(idx) == 2
    assert len(dists) == 2
    # Paris and London should be the two nearest
    assert set(idx) == {0, 2}


def test_geodesic_knn_k1():
    """k=1 returns the single nearest neighbor."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=1, metric="km")
    assert len(idx) == 1
    # Paris (~880 km) is closer than London (~930 km)
    assert idx[0] == 0


def test_geodesic_knn_all():
    """k=n returns all points sorted by distance."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=3, metric="km")
    assert len(idx) == 3
    # Distances should be sorted ascending
    assert list(dists) == sorted(dists)


def test_geodesic_knn_sorted():
    """Returned results are sorted nearest-first."""
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=3, metric="km")
    for i in range(len(dists) - 1):
        assert dists[i] <= dists[i + 1]


def test_geodesic_knn_distances_correct():
    """Returned distances should match individual geodist calls."""
    query = (52.5200, 13.4050)
    pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
    idx, dists = geodesic_knn(query, pts, k=3, metric="km")
    for i_pos, i_orig in enumerate(idx):
        expected = geodist(query, pts[i_orig], metric="km")
        assert dists[i_pos] == pytest.approx(expected, rel=1e-6)


def test_geodesic_knn_invalid_k_zero():
    """ValueError for k < 1."""
    with pytest.raises(ValueError):
        geodesic_knn((0.0, 0.0), [(1.0, 1.0)], k=0)


def test_geodesic_knn_invalid_k_too_large():
    """ValueError when k > number of candidates."""
    with pytest.raises(ValueError):
        geodesic_knn((0.0, 0.0), [(1.0, 1.0), (2.0, 2.0)], k=5)


def test_geodesic_knn_invalid_coords():
    """ValueError for invalid query point coordinates."""
    with pytest.raises(ValueError):
        geodesic_knn((95.0, 0.0), [(1.0, 1.0)], k=1)


def test_geodesic_knn_wrong_shape():
    """ValueError for candidates with wrong shape."""
    with pytest.raises(ValueError):
        geodesic_knn((0.0, 0.0), [(0.0,)], k=1)


def test_geodesic_knn_metric():
    """Verify metric parameter is applied correctly."""
    query = (52.5200, 13.4050)
    pts = [(48.8566, 2.3522)]
    _, d_m = geodesic_knn(query, pts, k=1, metric="meter")
    _, d_km = geodesic_knn(query, pts, k=1, metric="km")
    assert d_m[0] == pytest.approx(d_km[0] * 1000.0, rel=1e-8)


# ===========================================================================
# ELLIPSOID SUPPORT
# ===========================================================================


class TestResolveEllipsoid:
    """Tests for _resolve_ellipsoid helper."""

    def test_wgs84_default(self):
        a, f = _resolve_ellipsoid("WGS-84")
        assert a == pytest.approx(6378137.0)
        assert f == pytest.approx(1.0 / 298.257223563)

    def test_none_returns_wgs84(self):
        a, f = _resolve_ellipsoid(None)
        assert a == pytest.approx(6378137.0)

    def test_grs80(self):
        a, f = _resolve_ellipsoid("GRS-80")
        assert a == pytest.approx(6378137.0)
        assert f == pytest.approx(1.0 / 298.257222101)

    def test_custom_tuple(self):
        a, f = _resolve_ellipsoid((6377000.0, 1.0 / 300.0))
        assert a == pytest.approx(6377000.0)
        assert f == pytest.approx(1.0 / 300.0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ellipsoid"):
            _resolve_ellipsoid("INVALID-MODEL")

    def test_custom_tuple_negative_a_raises(self):
        with pytest.raises(ValueError, match="Semi-major axis"):
            _resolve_ellipsoid((-1.0, 1.0 / 298.0))

    def test_custom_tuple_zero_a_raises(self):
        with pytest.raises(ValueError, match="Semi-major axis"):
            _resolve_ellipsoid((0.0, 1.0 / 298.0))

    def test_custom_tuple_f_too_large_raises(self):
        with pytest.raises(ValueError, match="Flattening"):
            _resolve_ellipsoid((6378137.0, 1.0))

    def test_custom_tuple_f_negative_raises(self):
        with pytest.raises(ValueError, match="Flattening"):
            _resolve_ellipsoid((6378137.0, -0.01))

    def test_all_ellipsoids_resolve(self):
        for name in ELLIPSOIDS:
            a, _f = _resolve_ellipsoid(name)
            assert a > 6_000_000
            assert 0 < _f < 0.01


class TestEllipsoidGeodesic:
    """Tests for geodesic distance with different ellipsoids."""

    berlin = (52.5200, 13.4050)
    paris = (48.8566, 2.3522)

    def test_wgs84_matches_default(self):
        """Explicit WGS-84 must match the default (no ellipsoid)."""
        d_default = geodist(self.berlin, self.paris, metric="km")
        d_wgs84 = geodist(self.berlin, self.paris, metric="km", ellipsoid="WGS-84")
        assert d_default == pytest.approx(d_wgs84, rel=1e-12)

    def test_grs80_very_close_to_wgs84(self):
        """GRS-80 and WGS-84 differ by ~0.1mm in flattening → negligible distance diff."""
        d_wgs84 = geodist(self.berlin, self.paris, metric="meter", ellipsoid="WGS-84")
        d_grs80 = geodist(self.berlin, self.paris, metric="meter", ellipsoid="GRS-80")
        # Should be within 1 meter for ~880 km distance
        assert d_wgs84 == pytest.approx(d_grs80, abs=1.0)

    def test_different_ellipsoids_give_different_distances(self):
        """Clarke 1880 has noticeably different params → measurable distance diff."""
        d_wgs84 = geodist(self.berlin, self.paris, metric="meter", ellipsoid="WGS-84")
        d_clarke = geodist(
            self.berlin, self.paris, metric="meter", ellipsoid="Clarke (1880)"
        )
        # Should differ by more than 10m for ~880 km distance
        assert abs(d_wgs84 - d_clarke) > 10.0

    def test_custom_ellipsoid_tuple(self):
        """Custom (a, f) tuple should work."""
        d = geodist(
            self.berlin,
            self.paris,
            metric="km",
            ellipsoid=(6378137.0, 1.0 / 298.257223563),
        )
        d_wgs84 = geodist(self.berlin, self.paris, metric="km", ellipsoid="WGS-84")
        assert d == pytest.approx(d_wgs84, rel=1e-12)

    def test_intl_1924(self):
        """International 1924 ellipsoid should produce valid distances."""
        d = geodist(self.berlin, self.paris, metric="km", ellipsoid="Intl 1924")
        assert 870 < d < 890  # Berlin-Paris is ~879 km

    def test_all_named_ellipsoids_work(self):
        """All named ellipsoids should produce reasonable Berlin-Paris distances."""
        for name in ELLIPSOIDS:
            d = geodist(self.berlin, self.paris, metric="km", ellipsoid=name)
            assert 870 < d < 890, f"Ellipsoid {name} gave {d} km"


class TestEllipsoidBearing:
    """Tests for bearing with different ellipsoids."""

    def test_bearing_grs80(self):
        b = bearing((0.0, 0.0), (0.0, 1.0), ellipsoid="GRS-80")
        assert b == pytest.approx(90.0, abs=0.01)

    def test_bearing_ellipsoid_consistency(self):
        """Bearing should be very similar across close ellipsoids."""
        b_wgs = bearing((52.52, 13.405), (48.8566, 2.3522), ellipsoid="WGS-84")
        b_grs = bearing((52.52, 13.405), (48.8566, 2.3522), ellipsoid="GRS-80")
        assert b_wgs == pytest.approx(b_grs, abs=0.001)


class TestEllipsoidDestination:
    """Tests for destination with different ellipsoids."""

    def test_destination_roundtrip_grs80(self):
        a = (52.5200, 13.4050)
        b = (48.8566, 2.3522)
        b_deg = bearing(a, b, ellipsoid="GRS-80")
        d = geodist(a, b, metric="km", ellipsoid="GRS-80")
        lat, lon = destination(a, b_deg, d, metric="km", ellipsoid="GRS-80")
        assert lat == pytest.approx(b[0], abs=1e-3)
        assert lon == pytest.approx(b[1], abs=1e-3)


class TestEllipsoidInterpolate:
    """Tests for interpolate/midpoint with different ellipsoids."""

    def test_midpoint_grs80(self):
        mid = midpoint((0.0, 0.0), (0.0, 10.0), ellipsoid="GRS-80")
        assert mid[0] == pytest.approx(0.0, abs=0.01)
        assert mid[1] == pytest.approx(5.0, abs=0.05)

    def test_interpolate_clarke(self):
        pts = interpolate(
            (0.0, 0.0), (0.0, 10.0), n_points=4, ellipsoid="Clarke (1880)"
        )
        assert len(pts) == 4
        lons = [p[1] for p in pts]
        assert lons == sorted(lons)


class TestEllipsoidMatrix:
    """Tests for geodist_matrix with different ellipsoids."""

    def test_matrix_grs80_symmetric(self):
        coords = [(52.5200, 13.4050), (48.8566, 2.3522), (40.7128, -74.0060)]
        mat = geodist_matrix(coords, metric="km", ellipsoid="GRS-80")
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)


class TestEllipsoidSpatialQueries:
    """Tests for point_in_radius and geodesic_knn with different ellipsoids."""

    def test_point_in_radius_grs80(self):
        pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
        idx, _dists = point_in_radius(
            (52.5200, 13.4050), pts, 1000, metric="km", ellipsoid="GRS-80"
        )
        assert 0 in idx
        assert 2 in idx
        assert 1 not in idx

    def test_geodesic_knn_clarke(self):
        pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
        idx, _dists = geodesic_knn(
            (52.5200, 13.4050), pts, k=2, metric="km", ellipsoid="Clarke (1880)"
        )
        assert len(idx) == 2
        assert set(idx) == {0, 2}
