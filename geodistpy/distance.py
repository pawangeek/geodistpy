"""Computation of geospatial distances (WGS84).
# Computation of Geospatial Distances (WGS84)

Coordinates are assumed to be in Latitude and Longitude (WGS 84). Accepting numpy arrays as input.

The geospatial distance calculation is based on Vincenty's inverse method formula and accelerated with Numba (see `geokernels.geodesics.geodesic_vincenty` and references).

In a few cases (<0.01%) Vincenty's inverse method can fail to converge, and a fallback option using the slower geographiclib solution is implemented.

## Functions Included:

- `geodist`: returns a list of distances between points of two lists: `dist[i] = distance(XA[i], XB[i])`
- `geodist_matrix`: returns a distance matrix between all possible combinations of pairwise distances (either between all points in one list or points between two lists). `dist[i,j] = distance(XA[i], XB[j])` or `distance(X[i], X[j])`

This implementation provides a fast computation of geo-spatial distances in comparison to alternative methods for computing geodesic distance (tested: geopy and GeographicLib, see `geokernels.test_geodesics` for test functions).

## References:

- [Vincenty's Formulae](https://en.wikipedia.org/wiki/Vincenty's_formulae)
- [GeographicLib](https://geographiclib.sourceforge.io/)
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43–55. [arXiv:1109.4448](https://arxiv.org/abs/1109.4448). Bibcode:2013JGeod..87...43K. [doi:10.1007/s00190-012-0578-z](https://doi.org/10.1007/s00190-012-0578-z). Addenda.
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

from .geodesic import geodesic_vincenty, great_circle, great_circle_array


def _get_conv_factor(metric):
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


def geodist(coords1, coords2, metric="meter"):
    """
    Return distances between two coordinates or two lists of coordinates.

    Coordinates are assumed to be in Latitude, Longitude (WGS 84) format.

    For distances between all pair combinations, see geo_pdist and geo_cdist.

    Parameters:
        coords1 (array-like): The first set of coordinates in the format (latitude, longitude) or an array with shape (n_points1, 2) for multiple points.
        coords2 (array-like): The second set of coordinates in the format (latitude, longitude) or an array with shape (n_points2, 2) for multiple points.
            The shape of coords1 should match the shape of coords2.
        metric (str, optional): The unit of measurement for the calculated distances. Possible values are 'meter', 'km', 'mile', or 'nmi'.
            Default is 'meter'.

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

    if np.size(coords1) == 2:
        return geodesic_vincenty(coords1, coords2) * conv_fac
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
        [geodesic_vincenty(coords1[i], coords2[i]) for i in range(n_points)]
    )
    return dist * conv_fac


def geodist_matrix(coords1, coords2=None, metric="meter"):
    """
    Compute distance between each pair of possible combinations.

    If coords2 is None, compute distance between all possible pair combinations in coords1.
    dist[i, j] = distance(XA[i], XB[j])

    If coords2 is given, compute distance between each possible pair of the two collections
    of inputs: dist[i, j] = distance(X[i], X[j])

    Coordinates are assumed to be in Latitude, Longitude (WGS 84) format.

    Parameters:
        coords1 (list of tuples): List of coordinates in the format [(lat, long)] or an array with shape (n_points1, 2).
        coords2 (list of tuples, optional): List of coordinates in the format [(lat, long)] or an array with shape (n_points2, 2).
            If coords2 is not None, coords1.shape must match coords2.shape.
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

    if coords2 is None:
        dist = pdist(coords1, metric=lambda u, v: geodesic_vincenty(u, v))
        dist = squareform(dist)
    else:
        coords2 = np.asarray(coords2)

        # If two lists of coordinates are given
        assert coords1.shape == coords2.shape
        if (abs(coords2[:, 0]) > 90).any() or (abs(coords2[:, 1]) > 180).any():
            raise ValueError(
                "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
            )
        dist = cdist(coords1, coords2, metric=lambda u, v: geodesic_vincenty(u, v))
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
            If coords2 is not None, coords1.shape must match coords2.shape.
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
        dist = pdist(coords1, metric=lambda u, v: great_circle(u, v))
        dist = squareform(dist)
    else:
        coords2 = np.asarray(coords2)
        assert coords1.shape == coords2.shape

        if (abs(coords2[:, 0]) > 90).any() or (abs(coords2[:, 1]) > 180).any():
            raise ValueError(
                "Latitude values must be in the range [-90, 90] and Longitude values must be in the range [-180, 180]."
            )
        dist = cdist(coords1, coords2, metric=lambda u, v: great_circle(u, v))
    return dist * conv_fac
