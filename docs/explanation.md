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

## Conclusion

For applications that demand rapid and precise geospatial distance computations, Geodistpy is the clear choice. It offers exceptional speed improvements over both Geopy and Geographiclib, making it ideal for tasks involving large datasets or real-time geospatial applications. Despite its speed, Geodistpy maintains accuracy on par with Geographiclib, ensuring that fast calculations do not compromise precision.

By adopting Geodistpy, you can significantly enhance the efficiency and performance of your geospatial projects. It is a valuable tool for geospatial professionals and developers seeking both speed and accuracy in their distance computations.

To get started with Geodistpy, visit the [Geodistpy](https://github.com/pawangeek/geodistpy) and explore the documentation for comprehensive usage instructions.
