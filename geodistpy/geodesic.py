"""Geodesic distance calculation functions on a spheroid (WGS84).

The recommended function is based on Vincenty's inverse method formula
as implemented in the function geodesic_vincenty_inverse and accelerated with numba.

Alternative methods for computing geodesic distance via geopy or GeographicLib
are much slower (see README and test_geodesics.py).

However, in a few cases (<0.01%) Vincenty's inverse method can fail to converge, and
a fallback option using the slower geographiclib solution is implemented.


References:

- https://en.wikipedia.org/wiki/Vincenty's_formulae
- https://en.wikipedia.org/wiki/World_Geodetic_System
- https://en.wikipedia.org/wiki/Great-circle_distance
- https://geographiclib.sourceforge.io/
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43–55.
arXiv:1109.4448. Bibcode:2013JGeod..87...43K. doi:10.1007/s00190-012-0578-z. Addenda.

"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit

from geographiclib.geodesic import Geodesic as gglib


@jit(nopython=True)
def geodesic_vincenty_inverse(point1, point2):
    """
    Compute the geodesic distance between two points on the
    surface of a spheroid (WGS84) based on Vincenty's formula
    for the inverse geodetic problem.

    The function calculates the geodesic distance between two points on the Earth's surface
    using Vincenty's formula for the inverse geodetic problem. It is optimized with Numba's JIT
    (Just-In-Time) compilation for improved performance.

    Parameters:
        point1 : (latitude_1, longitude_1)
            The coordinates of the first point in the format (latitude, longitude) in degrees.
        point2 : (latitude_2, longitude_2)
            The coordinates of the second point in the format (latitude, longitude) in degrees.

    Returns:
        distance : float, in meters
            The geodesic distance between the points.

    Notes:
        - The function uses Vincenty's formula to compute the geodesic distance on the surface of a spheroid (WGS84).
        - It includes parameters for controlling the convergence of the iterative calculation.
        - The Earth's radius is assumed to be based on the WGS84 spheroid.
        - The function is optimized for performance using Numba's JIT compilation.

    Example:
        >>> point1 = (52.5200, 13.4050)
        >>> point2 = (48.8566, 2.3522)
        >>> distance = geodesic_vincenty_inverse(point1, point2)
        >>> distance
        878389.841013836

    Note:
        This function is an optimized implementation of the Vincenty python package
        (https://github.com/maurycyp/vincenty).
    """

    # WGS84 ellipsoid parameters:
    a = 6378137  # meters
    f = 1 / 298.257223563
    # b = (1 - f)a, in meters
    b = 6356752.314245

    # Inverse method parameters:
    max_iterations = 200
    convergence_threshold = 1e-11

    # Short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    u1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    u2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    l = math.radians(point2[1] - point1[1])
    lam = l

    sin_u1 = math.sin(u1)
    cos_u1 = math.cos(u1)
    sin_u2 = math.sin(u2)
    cos_u2 = math.cos(u2)

    for iteration in range(max_iterations):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )
        if sin_sigma == 0:
            return 0.0  # coincident points
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        if cos_sq_alpha != 0.0:
            cos_2sigma_m = cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha
            c = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        else:
            cos_2sigma_m = 0
            c = 0
        lam_prev = lam
        lam = l + (1 - c) * f * sin_alpha * (
            sigma
            + c
            * sin_sigma
            * (cos_2sigma_m + c * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )
        if abs(lam - lam_prev) < convergence_threshold:
            break
    else:
        # print('convergence', abs(lam - lam_prev)/ convergence_threshold)
        return None  # no convergence

    u_sq = cos_sq_alpha * (a**2 - b**2) / (b**2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                - B
                / 6
                * cos_2sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigma_m**2)
            )
        )
    )
    s = b * A * (sigma - delta_sigma)

    return round(s, 6)


def geodesic_vincenty(p1, p2):
    """
    Compute the geodesic distance between two points on the
    surface of a spheroid (WGS84) based on Vincenty's formula
    for the inverse geodetic problem[0].

    In the unlikely case Vincenty's inverse method fails to converge,
    the geographiclib algorithm is used instead.

    Parameters:
        p1 : (latitude_1, longitude_1)
            The coordinates of the first point in the format (latitude, longitude) in degrees.
        p2 : (latitude_2, longitude_2)
            The coordinates of the second point in the format (latitude, longitude) in degrees.

    Returns:
        distance : float, in meters
            The geodesic distance between the points.

    Notes:
        - The function calculates the geodesic distance on the surface of a spheroid (WGS84) using Vincenty's formula.
        - In case Vincenty's inverse method fails to converge, the geographiclib algorithm is used as a fallback.
        - The Earth's radius is assumed to be based on the WGS84 spheroid.

    Example:
        >>> p1 = (52.5200, 13.4050)
        >>> p2 = (48.8566, 2.3522)
        >>> distance = geodesic_vincenty(p1, p2)
        >>> distance
        878389.841013836
    """

    d = geodesic_vincenty_inverse(p1, p2)
    if d is None:
        # in case vincenty fails to converge, use geographiclib
        return gglib.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])["s12"]
    else:
        return d


def geodist_dimwise(X):
    """
    Compute the pairwise geodesic distances between data points for each dimension.

    The function calculates pairwise distances between data points along different dimensions.
    For the first two dimensions (latitude and longitude), it computes the combined geodesic distance
    as a single distance metric. For other dimensions, it calculates the pairwise Euclidean distances.

    Parameters:
        X (array-like, shape (n_samples, n_features)): An array representing data points.
            Each row corresponds to a data point, and each column represents a feature or dimension.
            The first two columns are assumed to contain latitude and longitude coordinates in degrees.

    Returns:
        distances (array-like, shape (n_samples, n_samples, n_features - 1)): An array containing pairwise distances
            between data points for each dimension. The distance for the first two dimensions (latitude and longitude)
            is computed as a combined geodesic distance and is represented in meters squared.
            Distances in other dimensions are pairwise Euclidean distances.

    Example:
        >>> data = np.array([
        ...     [52.5200, 13.4050, 100],
        ...     [48.8566, 2.3522, 200],
        ...     [40.7128, -74.0060, 300]
        ... ])
        >>> distances = geodist_dimwise(data)
        >>> distances.shape
        (3, 3, 2)  # Each element [i, j, k] represents the pairwise distance between data points i and j along dimension k.

    Notes:
        - The combined geodesic distance for the first two dimensions (latitude and longitude) is computed using the
          Vincenty formula.
        - For other dimensions beyond the first two, pairwise Euclidean distances are calculated.
    """

    # Initialise distances to zero
    dist = np.zeros((X.shape[0], X.shape[0], X.shape[1] - 1))
    # Compute geodesic distance for latitude and longitude
    dist[:, :, 0] = cdist(
        X[:, :2], X[:, :2], metric=lambda u, v: geodesic_vincenty(u, v)
    )
    # compute Euclidean distance for remaining dimensions
    dist[:, :, 1:] = X[:, np.newaxis, 2:] - X[np.newaxis, :, 2:]

    return dist


@jit(nopython=True)
def great_circle(u, v):
    """
    Use spherical geometry to calculate the surface distance between points.

    The function calculates the surface distance between two points on the Earth's surface
    using spherical geometry. It is optimized with Numba's JIT (Just-In-Time) compilation for
    improved performance.

    Parameters:
        u : (latitude_1, longitude_1)
            The coordinates of the first point in the format (latitude, longitude) in degrees.
        v : (latitude_2, longitude_2)
            The coordinates of the second point in the format (latitude, longitude) in degrees.

    Returns:
        distance : float, in meters
            The surface distance between the points.

    Notes:
        - The function uses spherical geometry to approximate the surface distance on the Earth's sphere.
        - The Earth's radius is assumed to be 6,371,009 meters.
        - The function is optimized for performance using Numba's JIT compilation.

    Example:
        >>> u = (52.5200, 13.4050)
        >>> v = (48.8566, 2.3522)
        >>> distance = great_circle(u, v)
        >>> distance
        878389.841013836
    """

    lat1, lng1 = math.radians(u[0]), math.radians(u[1])
    lat2, lng2 = math.radians(v[0]), math.radians(v[1])

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

    delta_lng = abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = math.cos(delta_lng), math.sin(delta_lng)

    d = math.atan2(
        math.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    return 6371009 * d


@jit(nopython=True)
def great_circle_array(u, v):
    """
    Use spherical geometry to calculate the surface distance between points.

    The function calculates the surface distance between two points on the Earth's surface
    using spherical geometry. It is optimized with Numba's JIT (Just-In-Time) compilation for
    improved performance.

    Parameters:
        u : (latitude_1, longitude_1), floats or arrays of floats
            The coordinates of the first point in the format (latitude, longitude) in degrees.
            It can be a single pair of coordinates or arrays of coordinates for multiple points.
        v : (latitude_2, longitude_2), floats or arrays of floats
            The coordinates of the second point in the format (latitude, longitude) in degrees.
            It can be a single pair of coordinates or arrays of coordinates for multiple points.

    Returns:
        distance : float or array of floats, in meters
            The surface distance between the points. If input arguments are arrays of coordinates,
            the result is an array of distances.

    Notes:
        - The function uses spherical geometry to approximate the surface distance on the Earth's sphere.
        - The Earth's radius is assumed to be 6,371,009 meters.
        - The function is optimized for performance using Numba's JIT compilation.

    Example:
        >>> u = (52.5200, 13.4050)
        >>> v = (48.8566, 2.3522)
        >>> distance = great_circle_array(u, v)
        >>> distance
        878389.841013836
    """

    lat1, lng1 = np.radians(u[0]), np.radians(u[1])
    lat2, lng2 = np.radians(v[0]), np.radians(v[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(
        np.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    return 6371009 * d


def geodist_dimwise_harvesine(X):
    """
    Compute the squared pairwise geodesic distances between data points for each dimension.

    The function calculates pairwise distances between data points along different dimensions.
    For latitude and longitude dimensions, it approximates the geodesic distances using the Haversine formula,
    and for other dimensions, it calculates squared Euclidean distances.

    Parameters:
        X (array-like, shape (n_samples, n_features)): An array representing data points.
            Each row corresponds to a data point, and each column represents a feature or dimension.
            The first two columns are assumed to contain latitude and longitude coordinates in degrees.

    Returns:
        distances (array-like, shape (n_samples, n_samples, n_features)): An array containing squared pairwise distances
            between data points for each dimension. The distances in the first two dimensions (latitude and longitude)
            are computed using the Haversine formula and are represented in meters squared.
            Distances in other dimensions are squared Euclidean distances.

    Notes:
        - Spherical geometry is used to approximate the surface distance with a mean earth radius of 6371.009 km.
        - The latitude and longitude dimensions are handled separately using the Haversine formula to account for
          the curvature of the Earth's surface.
        - For other dimensions beyond the first two, squared Euclidean distances are calculated.

    Example:
        >>> data = np.array([
        ...     [52.5200, 13.4050, 100],
        ...     [48.8566, 2.3522, 200],
        ...     [40.7128, -74.0060, 300]
        ... ])
        >>> distances = geodist_dimwise_haversine(data)
        >>> distances.shape
        (3, 3, 3)  # Each element [i, j, k] represents the squared distance between data points i and j along dimension k.
    """
    # Initialise distances to zero
    sdist = np.zeros((X.shape[0], X.shape[0], X.shape[1]))
    # Compute the haversine formula for latitude and longitude
    dlat = abs(np.radians(X[:, np.newaxis, 0] - X[np.newaxis, :, 0]))
    dlng = abs(np.radians(X[:, np.newaxis, 1] - X[np.newaxis, :, 1]))

    # delta latitude to meter
    sdist[:, :, 0] = (6371009 * 2 * np.arcsin(abs(np.sin(dlat / 2)))) ** 2

    # delta longitude to meter:
    sdist[:, :, 1] = (
        6371009
        * 2
        * np.arcsin(
            np.sqrt(
                (
                    1
                    - np.sin(dlat / 2) ** 2
                    - np.sin(np.radians(X[:, np.newaxis, 0] + X[np.newaxis, :, 0]) / 2)
                    ** 2
                )
                * np.sin(dlng / 2) ** 2
            )
        )
    ) ** 2
    # Compute the pairwise squared Euclidean distances between the data for any remaining dimensions
    sdist[:, :, 2:] = (X[:, np.newaxis, 2:] - X[np.newaxis, :, 2:]) ** 2

    return sdist
