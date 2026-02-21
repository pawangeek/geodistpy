---
title: Geodistpy – Fast Geodesic Distance Calculations in Python
description: Geodistpy is a blazing-fast Python library for geodesic distance calculations on the WGS84 ellipsoid. Up to 171x faster than Geopy with sub-millimeter accuracy.
---

# Geodistpy Documentation

Welcome to the documentation for the `geodistpy` project — a high-performance Python package for geospatial distance calculations between geographical coordinates.

## Table Of Contents

1. [Getting Started](getting-started.md) — Installation and quick start guide
2. [Bearing & Destination](bearing-destination.md) — Forward azimuth and Vincenty direct/inverse
3. [Interpolation & Midpoints](interpolation.md) — Geodesic waypoints and path generation
4. [Spatial Queries](spatial-queries.md) — k-NN and point-in-radius on the ellipsoid
5. [API Reference](api-reference.md) — Complete function reference
6. [Benchmarks & Internals](explanation.md) — Performance benchmarks and implementation details

## Introduction

The `geodistpy` package is a versatile library for geospatial calculations, offering various distance metrics and functionalities. It is built on the principles of geodesy and uses the WGS 84 coordinate system, making it suitable for a wide range of applications, from GPS tracking to geographical analysis.

### Key Features

- **Blazing fast:** ~0.4 µs per distance call — up to **171x faster than Geopy**, **109x faster than Geographiclib**
- **Parallel matrix computation:** Up to **1,230x faster** than Geopy for pairwise distance matrices using Numba parallel execution
- **Sub-millimeter accuracy:** Vincenty's inverse formula with mean error of just 9 µm vs Geographiclib reference
- **Improved Great Circle:** Andoyer-Lambert flattening correction reduces spherical approximation error by **700x** (from ~13 km to ~19 m)
- **Edge case handling:** Antipodal points, poles, date line crossings, and very short distances all handled correctly
- **Bearing & destination:** Compute forward azimuth between points and find destination given start + bearing + distance (Vincenty inverse/direct pair)
- **Geodesic interpolation:** Generate evenly-spaced waypoints and midpoints along geodesics for routing and visualization
- **Spatial queries:** Point-in-radius filtering (geofencing) and k-nearest-neighbour search using exact ellipsoidal distances

## Quick Example

```python
from geodistpy import geodist, bearing, destination, midpoint, geodesic_knn

# Define two coordinates (latitude, longitude)
berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

# Distance
distance_km = geodist(berlin, paris, metric='km')
print(f"Distance: {distance_km:.1f} km")

# Bearing (forward azimuth)
b = bearing(berlin, paris)
print(f"Bearing: {b:.2f}°")

# Destination: travel 500 km due east from Berlin
lat, lon = destination(berlin, 90.0, 500, metric='km')
print(f"Destination: ({lat:.4f}, {lon:.4f})")

# Midpoint
mid = midpoint(berlin, paris)
print(f"Midpoint: ({mid[0]:.4f}, {mid[1]:.4f})")

# k-NN: find 2 nearest cities to Berlin
cities = [(48.8566, 2.3522), (51.5074, -0.1278), (40.7128, -74.006)]
idx, dists = geodesic_knn(berlin, cities, k=2, metric='km')
print(f"2 nearest: {idx}, distances: {dists.round(1)} km")
```

## Links

- **PyPI:** [pypi.org/project/geodistpy](https://pypi.org/project/geodistpy/)
- **GitHub:** [github.com/pawangeek/geodistpy](https://github.com/pawangeek/geodistpy)

## License

The geodistpy project is licensed under the [MIT License](https://github.com/pawangeek/geodistpy/blob/main/LICENSE).
