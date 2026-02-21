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
- Karney, Charles F. F. (January 2013). "Algorithms for geodesics". Journal of Geodesy. 87 (1): 43-55.
arXiv:1109.4448. Bibcode:2013JGeod..87...43K. doi:10.1007/s00190-012-0578-z. Addenda.

"""

import math
import numpy as np
from numba import jit, prange

from geographiclib.geodesic import Geodesic as gglib


@jit(nopython=True, fastmath=True, cache=True)
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
    # b = (1 - f)a, in meters (full precision, not truncated)
    b = a * (1 - f)

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

    return s


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


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _vincenty_pdist(coords):
    """Compute pairwise Vincenty distances (pdist-style) using Numba parallel.

    Non-convergent pairs are marked with -1.0 as a sentinel value.
    Use vincenty_pdist() (without underscore) for automatic geographiclib fallback.
    """
    n = coords.shape[0]
    result = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            d = geodesic_vincenty_inverse(coords[i], coords[j])
            if d is None:
                d = -1.0  # sentinel: non-convergence
            result[i, j] = d
            result[j, i] = d
    return result


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _vincenty_cdist(coords1, coords2):
    """Compute cross-distance Vincenty matrix using Numba parallel.

    Non-convergent pairs are marked with -1.0 as a sentinel value.
    Use vincenty_cdist() (without underscore) for automatic geographiclib fallback.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    result = np.zeros((n1, n2))
    for i in prange(n1):
        for j in range(n2):
            d = geodesic_vincenty_inverse(coords1[i], coords2[j])
            if d is None:
                d = -1.0  # sentinel: non-convergence
            result[i, j] = d
    return result


def _apply_fallback(dist, coords1, coords2=None):
    """Replace sentinel values (-1.0) with geographiclib fallback distances.

    When Vincenty's method fails to converge (<0.01% of cases, typically
    near-antipodal points), the Numba functions mark those entries with -1.0.
    This function finds them and computes the correct distance via geographiclib.
    """
    mask = dist < 0.0
    if not mask.any():
        return dist
    indices = np.argwhere(mask)
    for idx in indices:
        i, j = idx[0], idx[1]
        p1 = coords1[i]
        p2 = coords2[j] if coords2 is not None else coords1[j]
        dist[i, j] = gglib.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])["s12"]
        # Mirror for symmetric pdist case
        if coords2 is None and i != j:
            dist[j, i] = dist[i, j]
    return dist


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _great_circle_pdist(coords):
    """Compute pairwise great circle distances (pdist-style) using Numba parallel."""
    n = coords.shape[0]
    result = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            d = great_circle(coords[i], coords[j])
            result[i, j] = d
            result[j, i] = d
    return result


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _great_circle_cdist(coords1, coords2):
    """Compute cross-distance great circle matrix using Numba parallel."""
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    result = np.zeros((n1, n2))
    for i in prange(n1):
        for j in range(n2):
            result[i, j] = great_circle(coords1[i], coords2[j])
    return result


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
    # Compute geodesic distance for latitude and longitude using Numba parallel
    dist[:, :, 0] = _vincenty_pdist(np.ascontiguousarray(X[:, :2]))
    # compute Euclidean distance for remaining dimensions
    dist[:, :, 1:] = X[:, np.newaxis, 2:] - X[np.newaxis, :, 2:]

    return dist


@jit(nopython=True, fastmath=True, cache=True)
def geodesic_vincenty_inverse_full(point1, point2):
    """
    Compute the geodesic distance **and** forward/back azimuths between two
    points on the WGS-84 ellipsoid using Vincenty's inverse formula.

    This is the "full" variant of :func:`geodesic_vincenty_inverse`; it
    returns a 3-element tuple ``(distance, fwd_azimuth, back_azimuth)``
    instead of just the scalar distance.

    Parameters:
        point1 : (latitude_1, longitude_1)
            The coordinates of the first point in degrees.
        point2 : (latitude_2, longitude_2)
            The coordinates of the second point in degrees.

    Returns:
        (distance, fwd_azimuth, back_azimuth) : (float, float, float)
            *distance* in **meters**, azimuths in **degrees** (0–360).
            Returns ``(0.0, 0.0, 0.0)`` for coincident points and
            ``(-1.0, 0.0, 0.0)`` when the iteration does not converge.
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)

    max_iterations = 200
    convergence_threshold = 1e-11

    if point1[0] == point2[0] and point1[1] == point2[1]:
        return (0.0, 0.0, 0.0)

    u1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    u2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    l = math.radians(point2[1] - point1[1])
    lam = l

    sin_u1 = math.sin(u1)
    cos_u1 = math.cos(u1)
    sin_u2 = math.sin(u2)
    cos_u2 = math.cos(u2)

    converged = False
    sin_lam = 0.0
    cos_lam = 0.0
    sin_sigma = 0.0
    cos_sigma = 0.0
    sigma = 0.0
    sin_alpha = 0.0
    cos_sq_alpha = 0.0
    cos_2sigma_m = 0.0

    for iteration in range(max_iterations):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )
        if sin_sigma == 0.0:
            return (0.0, 0.0, 0.0)
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha**2
        if cos_sq_alpha != 0.0:
            cos_2sigma_m = cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
        else:
            cos_2sigma_m = 0.0
            c = 0.0
        lam_prev = lam
        lam = l + (1.0 - c) * f * sin_alpha * (
            sigma
            + c
            * sin_sigma
            * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2))
        )
        if abs(lam - lam_prev) < convergence_threshold:
            converged = True
            break

    if not converged:
        return (-1.0, 0.0, 0.0)  # sentinel: non-convergence

    u_sq = cos_sq_alpha * (a**2 - b**2) / (b**2)
    A = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    B = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4.0
            * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)
                - B
                / 6.0
                * cos_2sigma_m
                * (-3.0 + 4.0 * sin_sigma**2)
                * (-3.0 + 4.0 * cos_2sigma_m**2)
            )
        )
    )
    s = b * A * (sigma - delta_sigma)

    # Forward azimuth (point1 → point2)
    fwd_az = math.degrees(
        math.atan2(
            cos_u2 * sin_lam,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam,
        )
    )
    # Back azimuth (point2 → point1)
    back_az = math.degrees(
        math.atan2(
            cos_u1 * sin_lam,
            -sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lam,
        )
    )
    # Normalise to [0, 360)
    fwd_az = fwd_az % 360.0
    back_az = back_az % 360.0

    return (s, fwd_az, back_az)


@jit(nopython=True, fastmath=True, cache=True)
def geodesic_vincenty_direct(point, azimuth_deg, distance_m):
    """
    Compute the destination point given a start point, initial bearing, and
    distance along the geodesic on the WGS-84 ellipsoid (Vincenty direct).

    Parameters:
        point : (latitude, longitude)
            Starting coordinates in degrees.
        azimuth_deg : float
            Initial bearing (forward azimuth) in degrees clockwise from north.
        distance_m : float
            Distance to travel along the geodesic in **meters**.

    Returns:
        (latitude, longitude) : (float, float)
            Destination point in degrees.

    References:
        Vincenty, T. (1975). "Direct and inverse solutions of geodesics on the
        ellipsoid with application of nested equations". Survey Review.
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)

    max_iterations = 200
    convergence_threshold = 1e-11

    alpha1 = math.radians(azimuth_deg)
    sin_alpha1 = math.sin(alpha1)
    cos_alpha1 = math.cos(alpha1)

    tan_u1 = (1.0 - f) * math.tan(math.radians(point[0]))
    cos_u1 = 1.0 / math.sqrt(1.0 + tan_u1**2)
    sin_u1 = tan_u1 * cos_u1

    sigma1 = math.atan2(tan_u1, cos_alpha1)
    sin_alpha = cos_u1 * sin_alpha1
    cos_sq_alpha = 1.0 - sin_alpha**2

    u_sq = cos_sq_alpha * (a**2 - b**2) / (b**2)
    A = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    B = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))

    sigma = distance_m / (b * A)

    for _iteration in range(max_iterations):
        cos_2sigma_m = math.cos(2.0 * sigma1 + sigma)
        sin_sigma = math.sin(sigma)
        cos_sigma = math.cos(sigma)

        delta_sigma = (
            B
            * sin_sigma
            * (
                cos_2sigma_m
                + B
                / 4.0
                * (
                    cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)
                    - B
                    / 6.0
                    * cos_2sigma_m
                    * (-3.0 + 4.0 * sin_sigma**2)
                    * (-3.0 + 4.0 * cos_2sigma_m**2)
                )
            )
        )
        sigma_prev = sigma
        sigma = distance_m / (b * A) + delta_sigma
        if abs(sigma - sigma_prev) < convergence_threshold:
            break

    sin_sigma = math.sin(sigma)
    cos_sigma = math.cos(sigma)
    cos_2sigma_m = math.cos(2.0 * sigma1 + sigma)

    lat2 = math.atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1,
        (1.0 - f)
        * math.sqrt(
            sin_alpha**2 + (sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1) ** 2
        ),
    )

    lam = math.atan2(
        sin_sigma * sin_alpha1,
        cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1,
    )

    c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
    L = lam - (1.0 - c) * f * sin_alpha * (
        sigma
        + c
        * sin_sigma
        * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2))
    )

    lon2 = math.radians(point[1]) + L

    return (math.degrees(lat2), math.degrees(lon2))


@jit(nopython=True, fastmath=True, cache=True)
def great_circle(u, v):
    """
    Calculate the surface distance between two points on the WGS84 ellipsoid
    using the great circle formula with an Andoyer-Lambert flattening correction.

    This provides a fast approximation that is significantly more accurate than
    a pure spherical great circle, by applying a first-order correction for
    the Earth's oblateness.

    Parameters:
        u : (latitude_1, longitude_1)
            The coordinates of the first point in the format (latitude, longitude) in degrees.
        v : (latitude_2, longitude_2)
            The coordinates of the second point in the format (latitude, longitude) in degrees.

    Returns:
        distance : float, in meters
            The surface distance between the points.

    Notes:
        - Uses the Vincenty special case for spherical central angle, then applies
          the Andoyer-Lambert correction for WGS84 flattening.
        - The function is optimized for performance using Numba's JIT compilation.

    References:
        - Andoyer, H. (1932). "Formula giving the length of the geodesic joining 2 points on the ellipsoid"
        - Lambert, W. D. (1942). "The distance between two widely separated points on the surface of the earth"

    Example:
        >>> u = (52.5200, 13.4050)
        >>> v = (48.8566, 2.3522)
        >>> distance = great_circle(u, v)
        >>> distance
        878389.841013836
    """

    # WGS84 parameters
    a = 6378137.0
    f = 1.0 / 298.257223563

    lat1, lng1 = math.radians(u[0]), math.radians(u[1])
    lat2, lng2 = math.radians(v[0]), math.radians(v[1])

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

    delta_lng = abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = math.cos(delta_lng), math.sin(delta_lng)

    # Spherical central angle (Vincenty formula for numerical stability)
    sigma = math.atan2(
        math.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    if sigma == 0.0:
        return 0.0

    # Andoyer-Lambert flattening correction
    F = (lat1 + lat2) / 2.0  # mean latitude
    G = (lat1 - lat2) / 2.0  # half latitude difference
    lam = delta_lng / 2.0  # half longitude difference

    sinF2 = math.sin(F) ** 2
    cosF2 = math.cos(F) ** 2
    sinG2 = math.sin(G) ** 2
    cosG2 = math.cos(G) ** 2
    sinL2 = math.sin(lam) ** 2
    cosL2 = math.cos(lam) ** 2

    S = sinG2 * cosL2 + cosF2 * sinL2
    C = cosG2 * cosL2 + sinF2 * sinL2

    omega = math.atan2(math.sqrt(S), math.sqrt(C))

    if omega == 0.0:
        return 0.0

    # Guard against division by zero (e.g., pole-to-pole where S or C == 0)
    if S == 0.0 or C == 0.0:
        return 2.0 * omega * a

    R = math.sqrt(S * C) / omega
    D = 2.0 * omega * a
    H1 = (3.0 * R - 1.0) / (2.0 * C)
    H2 = (3.0 * R + 1.0) / (2.0 * S)

    return D * (1.0 + f * (H1 * sinF2 * cosG2 - H2 * cosF2 * sinG2))


@jit(nopython=True)
def great_circle_array(u, v):
    """
    Calculate the surface distance between points on the WGS84 ellipsoid
    using the great circle formula with an Andoyer-Lambert flattening correction.

    Vectorized version that accepts arrays of coordinates.

    Parameters:
        u : (latitude_1, longitude_1), floats or arrays of floats
            The coordinates of the first point in the format (latitude, longitude) in degrees.
        v : (latitude_2, longitude_2), floats or arrays of floats
            The coordinates of the second point in the format (latitude, longitude) in degrees.

    Returns:
        distance : float or array of floats, in meters
            The surface distance between the points.

    Notes:
        - Uses the Vincenty special case for spherical central angle, then applies
          the Andoyer-Lambert correction for WGS84 flattening.
        - The function is optimized for performance using Numba's JIT compilation.

    Example:
        >>> u = (52.5200, 13.4050)
        >>> v = (48.8566, 2.3522)
        >>> distance = great_circle_array(u, v)
        >>> distance
        878389.841013836
    """

    # WGS84 parameters
    a = 6378137.0
    f = 1.0 / 298.257223563

    lat1, lng1 = np.radians(u[0]), np.radians(u[1])
    lat2, lng2 = np.radians(v[0]), np.radians(v[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = np.abs(lng2 - lng1)
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    # Spherical central angle (Vincenty formula for numerical stability)
    sigma = np.arctan2(
        np.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    # Andoyer-Lambert flattening correction
    F = (lat1 + lat2) / 2.0
    G = (lat1 - lat2) / 2.0
    lam = delta_lng / 2.0

    sinF2 = np.sin(F) ** 2
    cosF2 = np.cos(F) ** 2
    sinG2 = np.sin(G) ** 2
    cosG2 = np.cos(G) ** 2
    sinL2 = np.sin(lam) ** 2
    cosL2 = np.cos(lam) ** 2

    S = sinG2 * cosL2 + cosF2 * sinL2
    C = cosG2 * cosL2 + sinF2 * sinL2

    omega = np.arctan2(np.sqrt(S), np.sqrt(C))

    # Avoid division by zero for coincident points
    safe_omega = np.where(omega == 0.0, 1.0, omega)
    safe_S = np.where(S == 0.0, 1.0, S)
    safe_C = np.where(C == 0.0, 1.0, C)

    R = np.sqrt(S * C) / safe_omega
    D = 2.0 * omega * a
    H1 = (3.0 * R - 1.0) / (2.0 * safe_C)
    H2 = (3.0 * R + 1.0) / (2.0 * safe_S)

    result = D * (1.0 + f * (H1 * sinF2 * cosG2 - H2 * cosF2 * sinG2))

    # Zero out coincident points
    return np.where(sigma == 0.0, 0.0, result)


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
