---
title: Performance Benchmarks & Implementation – Geodistpy vs Geopy vs Geographiclib
description: Detailed benchmarks comparing geodistpy with Geopy and Geographiclib. Speed tests, accuracy analysis, edge cases, and implementation details of Vincenty and Great Circle methods.
---

# Explanation

Geodistpy is a powerful Python library designed for lightning-fast geospatial distance computations. It uses Vincenty's inverse formula accelerated with Numba JIT compilation and parallel execution to achieve extreme performance, while maintaining sub-millimeter accuracy on the WGS84 ellipsoid.

## Speed Comparison

```python
# Import libraries
from geopy.distance import geodesic as geodesic_geopy
from geographiclib.geodesic import Geodesic as geodesic_gglib
from geodistpy.geodesic import geodesic_vincenty

# Define two coordinates
coord1 = (52.5200, 13.4050)  # Berlin
coord2 = (48.8566, 2.3522)   # Paris

# Calculate distance with Geopy (based on Geographiclib)
distance_geopy = geodesic_geopy(coord1, coord2).meters

# Calculate distance with Geographiclib
distance_gglib = geodesic_gglib.WGS84.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])['s12']

# Calculate distance with Geodistpy
distance_geodistpy = geodesic_vincenty(coord1, coord2)

# Print the results
print(f"Distance between Berlin and Paris:")
print(f"Geopy: {distance_geopy} meters")
print(f"Geographiclib: {distance_gglib} meters")
print(f"Geodistpy: {distance_geodistpy} meters")
```

### Single-Pair Performance (10,000 calls, best of 3)

| Library | Total Time | Per Call |
|---|---|---|
| Geopy | ~636 ms | ~64 µs |
| Geographiclib | ~406 ms | ~41 µs |
| **Geodistpy (Vincenty+Numba)** | **~3.7 ms** | **~0.4 µs** |

- **Geodistpy is ~171x faster than Geopy**
- **Geodistpy is ~109x faster than Geographiclib**

### Pairwise Distance Matrix (N×N)

For matrix computations, Geodistpy uses Numba-parallel loops instead of scipy callbacks, yielding massive speedups:

| N points | Unique Pairs | Geopy | Geographiclib | Geodistpy | Speedup vs Geopy |
|---|---|---|---|---|---|
| 50 | 1,225 | 87 ms | 58 ms | **4.1 ms** | **21x** |
| 100 | 4,950 | 363 ms | 236 ms | **0.59 ms** | **613x** |
| 200 | 19,900 | 1.44 s | 930 ms | **1.17 ms** | **1,230x** |

### Accuracy (Geographiclib as reference, 5,000 random pairs)

| Method | Mean Error (m) | Max Error (m) | Mean Rel. Error |
|---|---|---|---|
| **Geodistpy (Vincenty)** | **0.000009** | **0.000108** | **1.03e-12** |
| Geopy (geodesic) | 0.000000 | 0.000000 | 4.40e-17 |
| **Geodistpy (Great Circle)** | **19.23** | **462.88** | **2.34e-06** |

> **Note:** Geopy shows zero error because it wraps Geographiclib — it's the same algorithm. Geodistpy's Vincenty mean error of **9 micrometers** is negligible. The Great Circle method uses an **Andoyer-Lambert flattening correction** that reduces error from ~13 km (pure sphere) to **~19 m** — a **700x improvement**.

### Great Circle vs Geodesic Trade-off

| Method | Time (10,000 calls) | Mean Error | Use Case |
|---|---|---|---|
| Great Circle + Andoyer-Lambert | 3.0 ms | ~19 m | Fast with good accuracy |
| Vincenty Geodesic | 4.9 ms | ~0.009 mm | Maximum precision |

## Performance Summary

- **Single-pair:** ~0.4 µs per call (171x faster than Geopy, 109x faster than Geographiclib)
- **Matrix (N=200):** 1.17 ms for 19,900 pairs (1,230x faster than Geopy)
- **Vincenty accuracy:** sub-millimeter (mean error = 9 µm)
- **Great Circle accuracy:** ~19 m mean error with Andoyer-Lambert flattening correction
- All edge cases handled: antipodal points, poles, date line crossings, short distances

## Key Implementation Details

### Vincenty's Inverse Formula
The core algorithm solves the inverse geodetic problem on the WGS84 ellipsoid using Vincenty's iterative method with 4th-order series expansions for the A and B Helmert coefficients. It is JIT-compiled with Numba using `fastmath=True` for additional floating-point optimizations.

### Andoyer-Lambert Flattening Correction
The Great Circle function applies a first-order correction for Earth's oblateness (WGS84 flattening f = 1/298.257223563). This dramatically improves accuracy compared to a pure spherical model while adding minimal computational cost.

### Numba Parallel Matrix Computation
Distance matrices use `@jit(parallel=True)` with `prange` for automatic multi-threaded execution, eliminating the overhead of Python-level callback functions that plague scipy-based approaches.

### Vincenty Direct Formula (Destination)
The `destination()` function implements Vincenty's direct formula, which is the complement to the inverse formula. Given a starting point, initial bearing, and distance, it computes the destination point on the WGS-84 ellipsoid. Both the inverse and direct Vincenty functions are JIT-compiled with Numba.

### Bearing (Forward Azimuth)
The `bearing()` function exposes the forward azimuth already computed internally by Vincenty's inverse iteration. A full inverse variant (`geodesic_vincenty_inverse_full`) returns (distance, forward_azimuth, back_azimuth) in a single pass, making bearing extraction essentially free.

### Geodesic Interpolation
The `interpolate()` and `midpoint()` functions combine the inverse and direct Vincenty formulas: first computing the total distance and azimuth via inverse, then stepping along the geodesic using direct to produce evenly-spaced waypoints. This is useful for route visualization and great-circle path rendering.

### Spatial Queries (k-NN and Point-in-Radius)
The `geodesic_knn()` and `point_in_radius()` functions fill the gap left by sklearn's `BallTree` which only supports haversine (spherical) distances. These functions use exact Vincenty ellipsoidal distances, providing higher accuracy for geofencing, store-locator, and spatial filtering applications.

### Multiple Ellipsoid Support
All distance, bearing, destination, interpolation, and spatial query functions accept an optional `ellipsoid` parameter. By default, WGS-84 is used. Six named ellipsoids are built in — **WGS-84**, **GRS-80**, **Airy (1830)**, **Intl 1924**, **Clarke (1880)**, and **GRS-67** — and users can also pass any custom `(semi_major_axis, flattening)` tuple for specialised geodetic applications. The `_resolve_ellipsoid()` helper converts the user input into `(a, f)` floats that are threaded through every Numba-JIT function, so switching ellipsoids incurs no additional overhead.

```python
from geodistpy import geodist, ELLIPSOIDS

# Named ellipsoid
d = geodist((52.52, 13.405), (48.8566, 2.3522), metric='km', ellipsoid='GRS-80')

# Custom (a, f) tuple
d = geodist((52.52, 13.405), (48.8566, 2.3522), ellipsoid=(6378137.0, 1/298.257223563))

# List available ellipsoids
print(ELLIPSOIDS.keys())
```

## Context and Background

The Python package `geodistpy` is a versatile library designed for geospatial calculations involving distances between geographical coordinates. It is built on the principles of geodesy and uses the WGS 84 coordinate system, which is commonly used in GPS and mapping applications.

## Why it was Created

The package was created to simplify and standardize geospatial distance calculations. Geographical distance calculations can be complex due to the curvature of the Earth's surface, and this library abstracts away those complexities, allowing users to focus on their specific geospatial tasks.

## Examples and Approaches

Let's explore multiple examples and approaches to working with the `geodistpy` library:

### Example 1: Calculating Distance Between Two Coordinates

```python
from geodistpy import geodist

# Define two coordinates in (latitude, longitude) format
coord1 = (52.5200, 13.4050)  # Berlin, Germany
coord2 = (48.8566, 2.3522)   # Paris, France

# Calculate the distance between the two coordinates in kilometers
distance_km = geodist(coord1, coord2, metric='km')
print(f"Distance between Berlin and Paris: {distance_km} kilometers")
```

### Example 2: Calculating Distance Between Multiple Coordinates

```python
from geodistpy import greatcircle_matrix

# Define a list of coordinates
coords = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]

# Calculate the distance matrix between all pairs of coordinates in miles
distance_matrix_miles = greatcircle_matrix(coords, metric='mile')
print("Distance matrix in miles:")
print(distance_matrix_miles)
```

### Example 3: Working with Different Metrics

The `geodistpy` library allows you to work with various distance metrics, such as meters, kilometers, miles, and nautical miles. You can easily switch between them by specifying the `metric` parameter.

```python
from geodistpy import geodist

coord1 = (52.5200, 13.4050)  # Berlin, Germany
coord2 = (48.8566, 2.3522)   # Paris, France

# Calculate the distance in meters
distance_meters = geodist(coord1, coord2, metric='meter')

# Calculate the distance in nautical miles
distance_nautical_miles = geodist(coord1, coord2, metric='nmi')

print(f"Distance in meters: {distance_meters}")
print(f"Distance in nautical miles: {distance_nautical_miles}")
```

### Example 4: Bearing and Destination

Compute the initial bearing from one point to another, or find where you arrive after travelling a given distance at a given bearing:

```python
from geodistpy import bearing, destination

berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

# Forward azimuth from Berlin to Paris
b = bearing(berlin, paris)
print(f"Bearing Berlin → Paris: {b:.2f}°")  # ~245.58°

# Travel 500 km due east from Berlin
lat, lon = destination(berlin, 90.0, 500, metric='km')
print(f"Destination: ({lat:.4f}, {lon:.4f})")
```

### Example 5: Geodesic Midpoint and Waypoints

Generate waypoints along a geodesic path — useful for route visualisation:

```python
from geodistpy import midpoint, interpolate

berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

# Geodesic midpoint
mid = midpoint(berlin, paris)
print(f"Midpoint: ({mid[0]:.4f}, {mid[1]:.4f})")

# 4 evenly-spaced interior waypoints
waypoints = interpolate(berlin, paris, n_points=4)
for i, wp in enumerate(waypoints, 1):
    print(f"  Waypoint {i}: ({wp[0]:.4f}, {wp[1]:.4f})")
```

### Example 6: Geofencing with Point-in-Radius

Find all points within a given geodesic radius:

```python
from geodistpy import point_in_radius

cities = [
    (48.8566, 2.3522),    # Paris
    (40.7128, -74.006),   # New York
    (51.5074, -0.1278),   # London
    (41.9028, 12.4964),   # Rome
]

# Which cities are within 1500 km of Berlin?
idx, dists = point_in_radius((52.5200, 13.4050), cities, 1500, metric='km')
print(f"Within 1500 km: indices {idx}, distances {dists.round(1)} km")
```

### Example 7: k-Nearest Neighbours on the Ellipsoid

Find the closest points using exact Vincenty distances (not haversine):

```python
from geodistpy import geodesic_knn

cities = [
    (48.8566, 2.3522),    # Paris
    (40.7128, -74.006),   # New York
    (51.5074, -0.1278),   # London
    (41.9028, 12.4964),   # Rome
]

idx, dists = geodesic_knn((52.5200, 13.4050), cities, k=2, metric='km')
print(f"2 nearest: indices {idx}, distances {dists.round(1)} km")
```

## Conclusion

For applications that demand rapid and precise geospatial distance computations, Geodistpy is the clear choice. It offers exceptional speed improvements over both Geopy and Geographiclib, making it ideal for tasks involving large datasets or real-time geospatial applications. Despite its speed, Geodistpy maintains accuracy on par with Geographiclib, ensuring that fast calculations do not compromise precision.

By adopting Geodistpy, you can significantly enhance the efficiency and performance of your geospatial projects. It is a valuable tool for geospatial professionals and developers seeking both speed and accuracy in their distance computations.

To get started with Geodistpy, visit the [Geodistpy](https://github.com/pawangeek/geodistpy) and explore the documentation for comprehensive usage instructions.
