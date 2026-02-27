"""Computation of geospatial distances (WGS84).
# Computation of Geospatial Distances (WGS84)

Coordinates are assumed to be in Latitude and Longitude (WGS 84). Accepting numpy arrays as input.

The geospatial distance calculation is based on Vincenty's inverse method formula
and accelerated with Numba (see `geodistpy.geodesic.geodesic_vincenty` and references).

In a few cases (<0.01%) Vincenty's inverse method can fail to converge, and a fallback option using the slower geographiclib solution is implemented.

## Functions Included:

- `geodist`: returns a list of distances between points of two lists: `dist[i] = distance(XA[i], XB[i])`
- `geodist_matrix`: returns a distance matrix between all possible combinations
  of pairwise distances (either between all points in one list or points between
  two lists). `dist[i,j] = distance(XA[i], XB[j])` or `distance(X[i], X[j])`

This implementation provides a fast computation of geo-spatial distances in comparison
to alternative methods for computing geodesic distance
(tested: geopy and GeographicLib, see `tests.test_geodist` for test functions).

## References:

- [Vincenty's Formulae](https://en.wikipedia.org/wiki/Vincenty's_formulae)
- [GeographicLib](https://geographiclib.sourceforge.io/)
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics".
  Journal of Geodesy. 87 (1): 43-55.
  [arXiv:1109.4448](https://arxiv.org/abs/1109.4448).
  [doi:10.1007/s00190-012-0578-z](https://doi.org/10.1007/s00190-012-0578-z).
"""

import math

import numpy as np

from .geodesic import (
    geodesic_vincenty,
    geodesic_vincenty_inverse_full,
    geodesic_vincenty_direct,
    great_circle,
    great_circle_array,
    _vincenty_pdist,
    _vincenty_cdist,
    _apply_fallback,
    _great_circle_pdist,
    _great_circle_cdist,
    _resolve_ellipsoid,
)

from geographiclib.geodesic import Geodesic as gglib


def _get_conv_factor(metric):
    """Return the conversion factor from meters to the given metric unit.

    Parameters:
        metric (str): Target unit. One of 'meter', 'km', 'mile', or 'nmi'.

    Returns:
        float: Multiplicative factor to convert meters to the target unit.

    Raises:
        ValueError: If the metric is not supported.
    """
    if metric == "meter":
        conv_fac = 1
    elif metric == "km":
        conv_fac = 1e-3
    elif metric == "mile":
        conv_fac = 1 / 1609.344
    elif metric == "nmi":
        conv_fac = 1 / 1852.0
    else:
        raise ValueError(f"Metric {metric} not supported")

    return conv_fac


def geodist(coords1, coords2, metric="meter", ellipsoid="WGS-84"):
    """
    Return distances between two coordinates or two lists of coordinates.

    Coordinates are assumed to be in Latitude, Longitude format.

    For distances between all pair combinations, see geo_pdist and geo_cdist.

    Parameters:
        coords1 (array-like): The first set of coordinates in the format (latitude, longitude) or an array with shape (n_points1, 2) for multiple points.
        coords2 (array-like): The second set of coordinates in the format (latitude, longitude) or an array with shape (n_points2, 2) for multiple points.
            The shape of coords1 should match the shape of coords2.
        metric (str, optional): The unit of measurement for the calculated distances. Possible values are 'meter', 'km', 'mile', or 'nmi'.
            Default is 'meter'.
        ellipsoid (str or tuple, optional): Ellipsoid model to use: a name
            from :data:`ELLIPSOIDS` (e.g. ``'WGS-84'``, ``'GRS-80'``) or a
            custom ``(a, f)`` tuple.  Default is ``'WGS-84'``.

    Returns:
        float or ndarray: The distance(s) between points, with a length of n_points.

    Raises:
        ValueError:
            - If the input coordinates do not have the expected shape.
            - If latitude values are not in the range [-90, 90].
            - If longitude values are not in the range [-180, 180].

    Examples:
        >>> geodist((52.5200, 13.4050), (48.8566, 2.3522), metric='km')
        878.389841013836

        >>> coords1 = [(37.7749, -122.4194), (34.0522, -118.2437)]
        >>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
        >>> geodist(coords1, coords2, metric='mile')
        array([2449.92107243, 1745.82567572])

        >>> geodist((37.7749, -122.4194), (37.7749, -122.4194))
        0.0
    """
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    assert coords1.shape == coords2.shape

    conv_fac = _get_conv_factor(metric)
    a, f = _resolve_ellipsoid(ellipsoid)

    if np.size(coords1) == 2:
        if coords1.shape[0] != 2 or coords2.shape[0] != 2:
            raise ValueError(
                "coords1 and coords2 must have two dimensions: Latitude, Longitude"
            )
        if (abs(coords1[0]) > 90).any() or (abs(coords2[0]) > 90).any():
            raise ValueError("Latitude values must be in the range [-90, 90]")
        if (abs(coords1[1]) > 180).any() or (abs(coords2[1]) > 180).any():
            raise ValueError("Longitude values must be in the range [-180, 180]")
        return geodesic_vincenty(coords1, coords2, a, f) * conv_fac

    if coords1.shape[1] != 2:
        raise ValueError(
            "coords1 and coords2 must have two dimensions: Latitude, Longitude"
        )
    if (abs(coords1[:, 0]) > 90).any() or (abs(coords2[:, 0]) > 90).any():
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if (abs(coords1[:, 1]) > 180).any() or (abs(coords2[:, 1]) > 180).any():
        raise ValueError("Longitude values must be in the range [-180, 180]")
    n_points = len(coords1)
    dist = np.asarray(
        [geodesic_vincenty(coords1[i], coords2[i], a, f) for i in range(n_points)]
    )
    return dist * conv_fac


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------
def bearing(point1, point2, ellipsoid="WGS-84"):
    """
    Compute the initial bearing (forward azimuth) from *point1* to *point2*
    on the selected ellipsoid using Vincenty's inverse formula.

    The bearing is measured clockwise from true north and returned in the
    range **[0, 360)** degrees.

    Parameters:
        point1 : tuple (latitude, longitude)
            Starting point in degrees.
        point2 : tuple (latitude, longitude)
            Destination point in degrees.
        ellipsoid : str or tuple, optional
            Ellipsoid model to use.  Default is ``'WGS-84'``.

    Returns:
        float
            Forward azimuth in degrees (0–360).

    Raises:
        ValueError: If latitude/longitude values are out of range.

    Examples:
        >>> bearing((52.5200, 13.4050), (48.8566, 2.3522))   # Berlin → Paris
        245.58...

        >>> bearing((0.0, 0.0), (0.0, 1.0))                  # due east on the equator
        90.0...
    """
    point1 = tuple(float(x) for x in point1)
    point2 = tuple(float(x) for x in point2)

    if abs(point1[0]) > 90 or abs(point2[0]) > 90:
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if abs(point1[1]) > 180 or abs(point2[1]) > 180:
        raise ValueError("Longitude values must be in the range [-180, 180]")

    a, f = _resolve_ellipsoid(ellipsoid)

    result = geodesic_vincenty_inverse_full(point1, point2, a, f)
    if result[0] < 0:
        # Vincenty failed to converge – fall back to geographiclib
        g = gglib(a, f).Inverse(point1[0], point1[1], point2[0], point2[1])
        return g["azi1"] % 360.0
    return result[1]


# ---------------------------------------------------------------------------
# Destination (Vincenty direct)
# ---------------------------------------------------------------------------
def destination(point, bearing_deg, distance, metric="meter", ellipsoid="WGS-84"):
    """
    Compute the destination point given a starting point, initial bearing,
    and distance along the geodesic on the selected ellipsoid (Vincenty
    direct).

    Parameters:
        point : tuple (latitude, longitude)
            Starting point in degrees.
        bearing_deg : float
            Initial bearing (forward azimuth) in degrees clockwise from north.
        distance : float
            Distance to travel in the unit specified by *metric*.
        metric : str, optional
            Unit of the *distance* parameter: ``'meter'``, ``'km'``,
            ``'mile'``, or ``'nmi'``.  Default is ``'meter'``.
        ellipsoid : str or tuple, optional
            Ellipsoid model to use.  Default is ``'WGS-84'``.

    Returns:
        tuple (latitude, longitude)
            Destination point in degrees.

    Raises:
        ValueError: If latitude/longitude values are out of range or metric
            is unsupported.

    Examples:
        >>> destination((52.5200, 13.4050), 245.0, 879.0, metric='km')
        (48.85..., 2.35...)

        >>> destination((0.0, 0.0), 90.0, 111.32, metric='km')
        (0.0, 1.0...)
    """
    point = tuple(float(x) for x in point)

    if abs(point[0]) > 90:
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if abs(point[1]) > 180:
        raise ValueError("Longitude values must be in the range [-180, 180]")

    conv_fac = _get_conv_factor(metric)
    distance_m = float(distance) / conv_fac  # convert to meters

    a, f = _resolve_ellipsoid(ellipsoid)

    lat, lon = geodesic_vincenty_direct(point, float(bearing_deg), distance_m, a, f)
    if math.isnan(lat):
        # Vincenty direct failed to converge – fall back to geographiclib
        g = gglib(a, f).Direct(point[0], point[1], float(bearing_deg), distance_m)
        lat, lon = g["lat2"], g["lon2"]
    # Normalise longitude to [-180, 180]
    lon = ((lon + 180.0) % 360.0) - 180.0
    return (lat, lon)


# ---------------------------------------------------------------------------
# Interpolation / midpoint along a geodesic
# ---------------------------------------------------------------------------
def interpolate(point1, point2, n_points=1, ellipsoid="WGS-84"):
    """
    Return evenly-spaced waypoints along the geodesic from *point1* to
    *point2* on the selected ellipsoid.

    When ``n_points=1`` the function returns the **midpoint**.  For
    ``n_points=N`` it returns *N* interior points that divide the geodesic
    into *N + 1* equal-length segments (the endpoints are **not** included).

    The implementation uses Vincenty's inverse formula to obtain the total
    distance and forward azimuth, then repeatedly applies Vincenty's direct
    formula to step along the geodesic.

    Parameters:
        point1 : tuple (latitude, longitude)
            Start point in degrees.
        point2 : tuple (latitude, longitude)
            End point in degrees.
        n_points : int, optional
            Number of interior waypoints to return.  Default is ``1``
            (midpoint only).
        ellipsoid : str or tuple, optional
            Ellipsoid model to use.  Default is ``'WGS-84'``.

    Returns:
        list of tuples [(lat, lon), ...]
            The waypoints in degrees, ordered from *point1* towards *point2*.
            The length of the list equals *n_points*.

    Raises:
        ValueError: If *n_points* < 1 or coordinates are out of range.

    Examples:
        >>> interpolate((0.0, 0.0), (0.0, 10.0), n_points=1)
        [(0.0, 5.0...)]

        >>> interpolate((0.0, 0.0), (0.0, 10.0), n_points=4)
        [(0.0, 2.0...), (0.0, 4.0...), (0.0, 6.0...), (0.0, 8.0...)]
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1")

    point1 = tuple(float(x) for x in point1)
    point2 = tuple(float(x) for x in point2)

    if abs(point1[0]) > 90 or abs(point2[0]) > 90:
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if abs(point1[1]) > 180 or abs(point2[1]) > 180:
        raise ValueError("Longitude values must be in the range [-180, 180]")

    a, f = _resolve_ellipsoid(ellipsoid)

    # Get total distance and forward azimuth via Vincenty inverse
    result = geodesic_vincenty_inverse_full(point1, point2, a, f)
    if result[0] < 0:
        # fallback to geographiclib
        g = gglib(a, f).Inverse(point1[0], point1[1], point2[0], point2[1])
        total_dist = g["s12"]
        fwd_az = g["azi1"]
    elif result[0] == 0.0:
        # Coincident points
        return [point1] * n_points
    else:
        total_dist = result[0]
        fwd_az = result[1]

    segment = total_dist / (n_points + 1)
    waypoints = []
    for i in range(1, n_points + 1):
        lat, lon = geodesic_vincenty_direct(point1, fwd_az, segment * i, a, f)
        if math.isnan(lat):
            # Vincenty direct failed to converge – fall back to geographiclib
            g = gglib(a, f).Direct(point1[0], point1[1], fwd_az, segment * i)
            lat, lon = g["lat2"], g["lon2"]
        lon = ((lon + 180.0) % 360.0) - 180.0
        waypoints.append((lat, lon))

    return waypoints


def midpoint(point1, point2, ellipsoid="WGS-84"):
    """
    Return the geodesic midpoint between two points on the given ellipsoid.

    This is a convenience wrapper around :func:`interpolate` with
    ``n_points=1``.

    Parameters:
        point1 : tuple (latitude, longitude)
            First point in degrees.
        point2 : tuple (latitude, longitude)
            Second point in degrees.
        ellipsoid : str or tuple, optional
            Ellipsoid model to use: a name from :data:`ELLIPSOIDS`
            (e.g. ``'WGS-84'``, ``'GRS-80'``) or a custom ``(a, f)``
            tuple.  Default is ``'WGS-84'``.

    Returns:
        tuple (latitude, longitude)
            Midpoint in degrees.

    Examples:
        >>> midpoint((0.0, 0.0), (0.0, 10.0))
        (0.0, 5.0...)
    """
    return interpolate(point1, point2, n_points=1, ellipsoid=ellipsoid)[0]


# ---------------------------------------------------------------------------
# Point-in-radius
# ---------------------------------------------------------------------------
def point_in_radius(center, candidates, radius, metric="meter", ellipsoid="WGS-84"):
    """
    Find all *candidate* points that lie within a given geodesic radius
    of a *center* point on the selected ellipsoid.

    Useful for geofencing, store-locator queries, and spatial filtering.

    Parameters:
        center : tuple (latitude, longitude)
            Reference point in degrees.
        candidates : array-like, shape (n, 2)
            Array of candidate points ``[(lat, lon), ...]``.
        radius : float
            Radius threshold in the unit specified by *metric*.
        metric : str, optional
            Unit for *radius*: ``'meter'``, ``'km'``, ``'mile'``, or
            ``'nmi'``.  Default is ``'meter'``.
        ellipsoid : str or tuple, optional
            Ellipsoid model to use.  Default is ``'WGS-84'``.

    Returns:
        tuple (indices, distances)
            *indices* : ndarray of int — indices into *candidates* of points
            that fall within the radius.
            *distances* : ndarray of float — the corresponding distances in
            the requested *metric*.

    Raises:
        ValueError: If coordinates are out of range, radius is negative,
            or metric is unsupported.

    Examples:
        >>> pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
        >>> idx, dists = point_in_radius((52.5200, 13.4050), pts, 1000, metric='km')
        >>> idx
        array([0, 2])
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")

    conv_fac = _get_conv_factor(metric)
    center = tuple(float(x) for x in center)

    if abs(center[0]) > 90:
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if abs(center[1]) > 180:
        raise ValueError("Longitude values must be in the range [-180, 180]")

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must have shape (n, 2)")

    # Compute distances from center to each candidate (in meters)
    center_arr = np.ascontiguousarray(
        np.tile(center, (len(candidates), 1)), dtype=np.float64
    )
    a, f = _resolve_ellipsoid(ellipsoid)

    dists_m = np.array(
        [
            geodesic_vincenty(center_arr[i], candidates[i], a, f)
            for i in range(len(candidates))
        ]
    )
    dists = dists_m * conv_fac

    mask = dists <= radius
    return np.where(mask)[0], dists[mask]


# ---------------------------------------------------------------------------
# k-Nearest Neighbours on geodesic distance
# ---------------------------------------------------------------------------
def geodesic_knn(point, candidates, k=1, metric="meter", ellipsoid="WGS-84"):
    """
    Find the *k* nearest neighbours to *point* among *candidates* using
    exact geodesic (Vincenty) distances on the selected ellipsoid.

    This fills the gap left by ``sklearn.neighbors.BallTree`` which only
    supports the haversine (spherical) metric.

    Parameters:
        point : tuple (latitude, longitude)
            Query point in degrees.
        candidates : array-like, shape (n, 2)
            Array of candidate points ``[(lat, lon), ...]``.
        k : int, optional
            Number of nearest neighbours to return.  Default is ``1``.
        metric : str, optional
            Unit for the returned distances: ``'meter'``, ``'km'``,
            ``'mile'``, or ``'nmi'``.  Default is ``'meter'``.
        ellipsoid : str or tuple, optional
            Ellipsoid model to use.  Default is ``'WGS-84'``.

    Returns:
        tuple (indices, distances)
            *indices* : ndarray of int, shape (k,) — indices into
            *candidates* of the *k* closest points, ordered nearest-first.
            *distances* : ndarray of float, shape (k,) — the corresponding
            distances in the requested *metric*.

    Raises:
        ValueError: If *k* < 1 or coordinates are out of range.

    Examples:
        >>> pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
        >>> idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=2, metric='km')
        >>> idx
        array([0, 2])
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    conv_fac = _get_conv_factor(metric)
    point = tuple(float(x) for x in point)

    if abs(point[0]) > 90:
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if abs(point[1]) > 180:
        raise ValueError("Longitude values must be in the range [-180, 180]")

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must have shape (n, 2)")

    n = len(candidates)
    if k > n:
        raise ValueError(f"k={k} is greater than the number of candidates ({n})")

    a, f = _resolve_ellipsoid(ellipsoid)

    dists_m = np.array(
        [geodesic_vincenty(point, tuple(candidates[i]), a, f) for i in range(n)]
    )
    dists = dists_m * conv_fac

    # Partial sort to find k smallest
    if k == n:
        order = np.argsort(dists)
    else:
        order = np.argpartition(dists, k)[:k]
        # Sort those k by distance
        order = order[np.argsort(dists[order])]

    return order, dists[order]


def geodist_matrix(coords1, coords2=None, metric="meter", ellipsoid="WGS-84"):
    """
    Compute distance between each pair of possible combinations.

    If coords2 is None, compute distance between all possible pair combinations in coords1.
    dist[i, j] = distance(XA[i], XB[j])

    If coords2 is given, compute distance between each possible pair of the two collections
    of inputs: dist[i, j] = distance(X[i], X[j])

    Coordinates are assumed to be in Latitude, Longitude format.

    Parameters:
        coords1 (list of tuples): List of coordinates in the format [(lat, long)] or an array with shape (n_points1, 2).
        coords2 (list of tuples, optional): List of coordinates in the format [(lat, long)] or an array with shape (n_points2, 2).
            Default is None.
        metric (str, optional): The unit of measurement for the calculated distances. Possible values are 'meter', 'km', 'mile', or 'nmi'.
            Default is 'meter'.
        ellipsoid (str or tuple, optional): Ellipsoid model to use: a name
            from :data:`ELLIPSOIDS` (e.g. ``'WGS-84'``, ``'GRS-80'``) or a
            custom ``(a, f)`` tuple.  Default is ``'WGS-84'``.

    Returns:
        ndarray: A distance matrix is returned.
            - If only coords1 is given, for each i and j, the metric dist(u=XA[i], v=XA[j]) is computed.
            - If coords2 is not None, for each i and j, the metric dist(u=XA[i], v=XB[j]) is computed and stored in the ij-th entry.

    Raises:
        ValueError:
            - If the input coordinates do not have the expected shape.
            - If latitude values are not in the range [-90, 90].
            - If longitude values are not in the range [-180, 180].

    Examples:
        >>> coords1 = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
        >>> geodist_matrix(coords1, metric='km')
        array([[   0.        ,  878.38984101, 8786.58652276],
               [ 878.38984101,    0.        , 9525.03650888],
               [8786.58652276, 9525.03650888,    0.        ]])

        >>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
        >>> geodist_matrix(coords1, coords2, metric='mile')
        array([[ 3060.81391478, 2437.78157493],
               [ 4290.62813902, 1745.82567572],
               [ 2449.92107243, 1746.57308007]])
    """
    conv_fac = _get_conv_factor(metric)

    coords1 = np.asarray(coords1)
    if coords1.shape[1] != 2:
        raise ValueError(
            "coords1 and coords2 must have two dimensions: Latitude, Longitude"
        )
    if (abs(coords1[:, 0]) > 90).any() or (abs(coords1[:, 1]) > 180).any():
        raise ValueError(
            "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
        )

    a, f = _resolve_ellipsoid(ellipsoid)

    if coords2 is None:
        dist = _vincenty_pdist(np.ascontiguousarray(coords1, dtype=np.float64), a, f)
        dist = _apply_fallback(dist, coords1, a=a, f=f)
    else:
        coords2 = np.asarray(coords2)

        if coords2.ndim != 2 or coords2.shape[1] != 2:
            raise ValueError(
                "coords1 and coords2 must have two dimensions: Latitude, Longitude"
            )
        if (abs(coords2[:, 0]) > 90).any() or (abs(coords2[:, 1]) > 180).any():
            raise ValueError(
                "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
            )
        dist = _vincenty_cdist(
            np.ascontiguousarray(coords1, dtype=np.float64),
            np.ascontiguousarray(coords2, dtype=np.float64),
            a,
            f,
        )
        dist = _apply_fallback(dist, coords1, coords2, a=a, f=f)
    return dist * conv_fac


def greatcircle(coords1, coords2, metric="meter"):
    """Calculate the distance between two sets of coordinates using the Great Circle approximation.

    Args:
        coords1 (array-like): The first set of coordinates in the format (latitude, longitude) or an array with shape (n_points1, 2) for multiple points.
        coords2 (array-like): The second set of coordinates in the format (latitude, longitude) or an array with shape (n_points2, 2) for multiple points.
            The shape of coords1 should match the shape of coords2.
        metric (str, optional): The unit of measurement for the calculated distances. Possible values are 'meter', 'km', 'mile', or 'nmi'.
            Default is 'meter'.

    Returns:
        float or ndarray: The distance(s) between the points. If multiple points are provided, an ndarray is returned.

    Raises:
        ValueError:
            - If the input coordinates do not have the expected shape.
            - If latitude values are not in the range [-90, 90].
            - If longitude values are not in the range [-180, 180].

    Notes:
        The Great Circle formula assumes a spherical Earth and may not be completely accurate
        for very long distances or in regions with significant variation in the Earth's curvature.

    Examples:
        >>> greatcircle((52.5200, 13.4050), (48.8566, 2.3522), metric='km')
        878.389841013836

        >>> coords1 = [(37.7749, -122.4194), (34.0522, -118.2437)]
        >>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
        >>> greatcircle(coords1, coords2, metric='mile')
        array([2449.92107243, 1745.82567572])

        >>> greatcircle((37.7749, -122.4194), (37.7749, -122.4194))
        0.0
    """
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    assert coords1.shape == coords2.shape

    conv_fac = _get_conv_factor(metric)

    if np.size(coords1) == 2:
        return great_circle_array(coords1, coords2) * conv_fac
    if coords1.shape[1] != 2:
        raise ValueError(
            "coords1 and coords2 must have two dimensions: Latitude, Longitude"
        )
    if (abs(coords1[:, 0]) > 90).any() or (abs(coords2[:, 0]) > 90).any():
        raise ValueError("Latitude values must be in the range [-90, 90]")
    if (abs(coords1[:, 1]) > 180).any() or (abs(coords2[:, 1]) > 180).any():
        raise ValueError("Longitude values must be in the range [-180, 180]")

    dist = great_circle_array(coords1, coords2)
    return dist * conv_fac


def greatcircle_matrix(coords1, coords2=None, metric="meter"):
    """
    Compute distance between each pair of possible combinations
    using spherical asymmetry (Great Circle approximation).

    If coords2 is None, compute distance between all possible pair combinations in coords1.
    dist[i, j] = distance(XA[i], XB[j])

    If coords2 is given, compute distance between each possible pair of the two collections
    of inputs: dist[i, j] = distance(X[i], X[j])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84) format.

    Parameters:
        coords1 (list of tuples): List of coordinates in the format [(lat, long)] or an array with shape (n_points1, 2).
        coords2 (list of tuples, optional): List of coordinates in the format [(lat, long)] or an array with shape (n_points2, 2).
            Default is None.
        metric (str, optional): The unit of measurement for the calculated distances. Possible values are 'meter', 'km', 'mile', or 'nmi'.
            Default is 'meter'.

    Returns:
        ndarray: A distance matrix is returned.
            - If only coords1 is given, for each i and j, the metric dist(u=XA[i], v=XA[j]) is computed.
            - If coords2 is not None, for each i and j, the metric dist(u=XA[i], v=XB[j]) is computed and stored in the ij-th entry.

    Raises:
        ValueError:
            - If the input coordinates do not have the expected shape.
            - If latitude values are not in the range [-90, 90].
            - If longitude values are not in the range [-180, 180].

    Examples:
        >>> coords1 = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
        >>> greatcircle_matrix(coords1, metric='km')
        array([[   0.        ,  878.38984101, 8786.58652276],
               [ 878.38984101,    0.        , 9525.03650888],
               [8786.58652276, 9525.03650888,    0.        ]])

        >>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
        >>> greatcircle_matrix(coords1, coords2, metric='mile')
        array([[ 3060.81391478, 2437.78157493],
               [ 4290.62813902, 1745.82567572],
               [ 2449.92107243, 1746.57308007]])
    """
    conv_fac = _get_conv_factor(metric)

    coords1 = np.asarray(coords1)
    if coords1.shape[1] != 2:
        raise ValueError(
            "coords1 and coords2 must have two dimensions: Latitude, Longitude"
        )
    if (abs(coords1[:, 0]) > 90).any() or (abs(coords1[:, 1]) > 180).any():
        raise ValueError(
            "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
        )

    if coords2 is None:
        # If only one list of coordinates is given:
        dist = _great_circle_pdist(np.ascontiguousarray(coords1, dtype=np.float64))
    else:
        coords2 = np.asarray(coords2)

        if coords2.ndim != 2 or coords2.shape[1] != 2:
            raise ValueError(
                "coords1 and coords2 must have two dimensions: Latitude, Longitude"
            )
        if (abs(coords2[:, 0]) > 90).any() or (abs(coords2[:, 1]) > 180).any():
            raise ValueError(
                "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
            )
        dist = _great_circle_cdist(
            np.ascontiguousarray(coords1, dtype=np.float64),
            np.ascontiguousarray(coords2, dtype=np.float64),
        )
    return dist * conv_fac
