---
title: Geodistpy – Fast Geodesic Distance Calculations in Python
description: Geodistpy is a blazing-fast Python library for geodesic distance calculations on the WGS84 ellipsoid. Up to 171x faster than Geopy with sub-millimeter accuracy.
---

# Geodistpy Documentation

Welcome to the documentation for the `geodistpy` project — a high-performance Python package for geospatial distance calculations between geographical coordinates.

## Table Of Contents

1. [Getting Started](getting-started.md) — Installation and quick start guide
2. [Explanation](explanation.md) — Benchmarks, accuracy, and implementation details
3. [API Reference](api-reference.md) — Complete function reference

## Introduction

The `geodistpy` package is a versatile library for geospatial calculations, offering various distance metrics and functionalities. It is built on the principles of geodesy and uses the WGS 84 coordinate system, making it suitable for a wide range of applications, from GPS tracking to geographical analysis.

### Key Features

- **Blazing fast:** ~0.4 µs per distance call — up to **171x faster than Geopy**, **109x faster than Geographiclib**
- **Parallel matrix computation:** Up to **1,230x faster** than Geopy for pairwise distance matrices using Numba parallel execution
- **Sub-millimeter accuracy:** Vincenty's inverse formula with mean error of just 9 µm vs Geographiclib reference
- **Improved Great Circle:** Andoyer-Lambert flattening correction reduces spherical approximation error by **700x** (from ~13 km to ~19 m)
- **Edge case handling:** Antipodal points, poles, date line crossings, and very short distances all handled correctly

## Quick Example

```python
from geodistpy import geodist

# Define two coordinates (latitude, longitude)
coord1 = (52.5200, 13.4050)  # Berlin, Germany
coord2 = (48.8566, 2.3522)   # Paris, France

distance_km = geodist(coord1, coord2, metric='km')
print(f"Distance: {distance_km} km")
```

## Links

- **PyPI:** [pypi.org/project/geodistpy](https://pypi.org/project/geodistpy/)
- **GitHub:** [github.com/pawangeek/geodistpy](https://github.com/pawangeek/geodistpy)

## License

The geodistpy project is licensed under the [MIT License](https://github.com/pawangeek/geodistpy/blob/main/LICENSE).
