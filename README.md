# Geodistpy: Fast and Accurate Geospatial Distance Computations

[![pypi](https://img.shields.io/pypi/v/geodistpy?label=PyPI&logo=PyPI&logoColor=white&color=blue)](https://pypi.python.org/pypi/geodistpy)
[![lint](https://github.com/pawangeek/geodistpy/actions/workflows/lint.yml/badge.svg)](https://github.com/pawangeek/geodistpy/actions/workflows/lint.yml)
[![Build](https://github.com/pawangeek/geodistpy/actions/workflows/build_package.yml/badge.svg)](https://github.com/pawangeek/geodistpy/actions/workflows/build_package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geodistpy?label=Python&logo=Python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Geodistpy is a powerful Python library designed for lightning-fast geospatial distance computations. In this README, we'll compare Geodistpy with three other popular libraries, Geopy, Geographiclib, and Pyproj, to highlight the significant performance advantages of Geodistpy.

**Features:** Vincenty geodesic (sub-millimeter accuracy) · Great circle with Andoyer-Lambert · Bearing & destination · Geodesic interpolation & midpoint · k-NN and point-in-radius · **Pandas & GeoPandas support** (pass DataFrames directly; optional `[pandas]` / `[geopandas]` extras)

* Documentation: https://pawangeek.github.io/geodistpy/
* Github Repo: https://github.com/pawangeek/geodistpy
* PyPI: https://pypi.org/project/geodistpy/

## Installation

```bash
pip install geodistpy
```

Optional extras for **pandas** and **GeoPandas** (DataFrame/GeoDataFrame support for `geodist_to_many`, `geodesic_knn`, `point_in_radius`, and `coordinates_from_df`):

```bash
pip install geodistpy[pandas]       # pandas DataFrame
pip install geodistpy[geopandas]    # GeoPandas GeoDataFrame (includes pandas)
```

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

We conducted a thorough benchmark comparing Geodistpy, Geopy, Geographiclib, and Pyproj across multiple scenarios: single-pair calls, pairwise distance matrices, accuracy analysis, edge cases, and scaling tests.

> Latest benchmark snapshot (from `poetry run python benchmark.py`, Mar 14, 2026).

### Test 1: Single Pair Distance (10,000 calls, best of 3)

| Library | Total Time | Per Call |
|---|---|---|
| Geopy | 619 ms | ~62 µs |
| Geographiclib | 396 ms | ~40 µs |
| Pyproj (`Geod.inv`) | 5.42 ms | ~0.5 µs |
| **Geodistpy (Vincenty+Numba)** | **3.78 ms** | **~0.4 µs** |

- **Geodistpy is 164x faster than Geopy**
- **Geodistpy is 105x faster than Geographiclib**
- **Geodistpy is 1.4x faster than Pyproj**

### Test 2: Pairwise Distance Matrix (N×N)

Geodistpy uses Numba-parallel loops (`prange`) instead of scipy callbacks, yielding massive speedups for matrix operations:

| N (points) | Unique Pairs | Geopy | Geographiclib | Pyproj | Geodistpy | vs Geopy |
|---|---|---|---|---|---|---|
| 50 | 1,225 | 83 ms | 56 ms | 0.91 ms | **4.85 ms** | **17x** |
| 100 | 4,950 | 339 ms | 231 ms | 3.60 ms | **0.48 ms** | **702x** |
| 200 | 19,900 | 1.43 s | 930 ms | 14.59 ms | **1.21 ms** | **1,183x** |

### Test 3: Accuracy (Geographiclib as reference, 5,000 random pairs)

| Method | Mean Error (m) | Max Error (m) | Mean Rel. Error | Max Rel. Error |
|---|---|---|---|---|
| **Geodistpy (Vincenty)** | **0.000009** | **0.000108** | **1.03e-12** | **1.24e-10** |
| Geopy (geodesic) | 0.000000 | 0.000000 | 4.40e-17 | 2.26e-16 |
| Pyproj (`Geod.inv`) | 0.000000 | 0.000000 | 4.72e-17 | 4.92e-15 |
| **Geodistpy (Great Circle)** | **19.23** | **462.88** | **2.34e-06** | **2.40e-05** |

> **Note on error values:** Geopy shows zero error because it is a direct wrapper around Geographiclib — comparing it to Geographiclib is comparing the same algorithm to itself. Geodistpy's Vincenty mean error of **0.000009m (9 micrometers)** is a negligible difference arising from using a different geodesic algorithm (Vincenty's inverse vs Karney's). This is far below any practical GPS or measurement precision. The Great Circle method uses an **Andoyer-Lambert flattening correction** on top of the spherical formula, reducing error from ~13 km (pure sphere) to **~19 m** — a **700x improvement** while remaining nearly as fast.

Geodistpy's Vincenty implementation maintains **sub-millimeter accuracy** (mean error = 9 µm) while being orders of magnitude faster. The Great Circle method now achieves **~19 m mean accuracy** — suitable for most practical applications.

### Test 4: Edge Cases

| Scenario | Geographiclib (m) | Geodistpy (m) | Δ Error (m) |
|---|---|---|---|
| Same point | 0.000 | 0.000 | 0.000000 |
| North Pole → South Pole | 20,003,931.459 | 20,003,931.459 | 0.000002 |
| Antipodal (equator) | 20,003,931.459 | 20,003,931.459 | 0.000000 |
| Near-antipodal | 20,003,008.422 | 20,003,008.422 | 0.000000 |
| Very short (~1m) | 1.113 | 1.113 | 0.000000 |
| Cross date line | 22,263.898 | 22,263.898 | 0.000003 |
| High latitude (Arctic) | 2,233.880 | 2,233.880 | 0.000000 |
| Sydney → New York | 15,988,007.485 | 15,988,007.485 | 0.000041 |
| London → Tokyo | 9,582,151.069 | 9,582,151.069 | 0.000018 |

All edge cases — including antipodal points, poles, date line crossings, and very short distances — are handled correctly with sub-millimeter precision.

### Test 5: Scaling (Sequential point-to-point calls)

| N calls | Geodistpy | Geographiclib | Pyproj | Geodistpy vs Pyproj |
|---|---|---|---|---|
| 1,000 | 0.55 ms | 45 ms | 0.74 ms | **1.3x faster** |
| 10,000 | 5.40 ms | 460 ms | 7.44 ms | **1.4x faster** |
| 50,000 | 26.1 ms | *(skipped)* | *(skipped)* | — |

### Test 6: Great Circle vs Geodesic Trade-off

| Method | Time (10,000 calls) | Mean Error | Use Case |
|---|---|---|---|
| Great Circle + Andoyer-Lambert (Numba) | 2.99 ms | **~19 m** | Fast with good accuracy |
| Vincenty Geodesic (Numba) | 4.93 ms | ~0.009 mm | Maximum precision |

Great Circle with Andoyer-Lambert correction is **1.6x faster** than Vincenty with only **~19 m average error** — a **700x improvement** over the previous pure spherical approach (~13.5 km error).

## Performance Summary

- **Single-pair:** Geodistpy is **164x faster than Geopy**, **105x faster than Geographiclib**, and **1.4x faster than Pyproj** (~0.4 µs/call).
- **Matrix (N=200):** Geodistpy is **1,183x faster than Geopy**, **770x faster than Geographiclib**, and **12.1x faster than Pyproj** (1.21 ms for 19,900 pairs).
- Vincenty: **sub-millimeter accuracy** (mean error = 9 µm vs Geographiclib reference).
- Great Circle: **~19 m mean accuracy** with Andoyer-Lambert flattening correction (700x better than pure sphere).
- All edge cases handled correctly: antipodal points, poles, date line, short distances.
- Per-call cost: **~0.4 µs** (Geodistpy) vs ~0.5 µs (Pyproj) vs ~40 µs (Geographiclib) vs ~62 µs (Geopy).

## Context and Background

The Python package `geodistpy` is a versatile library designed for geospatial calculations involving distances between geographical coordinates. It is built on the principles of geodesy and uses the WGS 84 coordinate system, which is commonly used in GPS and mapping applications.

## Why it was Created

The package was created to simplify and standardize geospatial distance calculations. Geographical distance calculations can be complex due to the curvature of the Earth's surface, and this library abstracts away those complexities, allowing users to focus on their specific geospatial tasks.

## After geocoding: distance and nearest-neighbor

Geocoding (address → lat/lon) is often the first step—with [geopy](https://geopy.readthedocs.io/) or any other geocoder. The next step is usually **"How far from this user to these N stores?"** or **"Which store is nearest?"** Geodistpy is built for that: once you have coordinates, do all distance and nearest-neighbor work **fast** with geodesic accuracy.

```python
# 1. Get coordinates (e.g. from geopy, your API, or a database)
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my_app")
user_location = geolocator.geocode("Alexanderplatz, Berlin")
user = (user_location.latitude, user_location.longitude)

# 2. Your store locations (lat, lon)
stores = [
    (52.5200, 13.4050),   # Berlin
    (48.8566, 2.3522),    # Paris
    (51.5074, -0.1278),   # London
]

# 3. How far from this user to each store? (one-to-many)
from geodistpy import geodist_to_many
distances_km = geodist_to_many(user, stores, metric="km")

# 4. Which store is nearest? (k-NN)
from geodistpy import geodesic_knn
nearest_idx, nearest_dists = geodesic_knn(user, stores, k=1, metric="km")

# 5. Which stores are within 500 km? (point-in-radius)
from geodistpy import point_in_radius
within_idx, within_dists = point_in_radius(user, stores, 500, metric="km")
```

No projection, no haversine hacks—just fast geodesic (Vincenty) on WGS84.

## Pandas / GeoPandas support

You can pass **pandas DataFrames** or **GeoPandas GeoDataFrames** directly to the functions that take points or candidates. Install the optional dependency:

```bash
pip install geodistpy[pandas]      # for pandas DataFrame
pip install geodistpy[geopandas]    # for GeoDataFrame (includes pandas)
```

- **DataFrame:** Use columns named `lat`/`lon` or `latitude`/`longitude`, or pass `lat_col` and `lon_col`.
- **GeoDataFrame:** Point geometry is used (WGS84: x=lon, y=lat).

**Example with pandas:**

```python
import pandas as pd
from geodistpy import geodist_to_many, geodesic_knn, point_in_radius, coordinates_from_df

# DataFrame with lat/lon columns
stores_df = pd.DataFrame({
    "lat": [48.8566, 51.5074, 40.7128],
    "lon": [2.3522, -0.1278, -74.006],
    "name": ["Paris", "London", "New York"],
})

# One-to-many distances — returns a Series indexed like your DataFrame
user = (52.52, 13.40)
distances = geodist_to_many(user, stores_df, metric="km")

# k nearest (pass DataFrame; get back indices and distances)
idx, dists = geodesic_knn(user, stores_df, k=2, metric="km")
nearest_stores = stores_df.iloc[idx]

# All stores within 1000 km
idx, dists = point_in_radius(user, stores_df, 1000, metric="km")
within = stores_df.loc[idx]
```

**Extract coordinates only** (e.g. for `geodist_matrix` or custom use):

```python
coords, index = coordinates_from_df(stores_df)
# coords is (n, 2) array of (lat, lon); index is stores_df.index
```

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

### Example 4: Computing Bearing Between Two Points

```python
from geodistpy import bearing

# Initial bearing from Berlin to Paris (clockwise from north)
b = bearing((52.5200, 13.4050), (48.8566, 2.3522))
print(f"Bearing from Berlin to Paris: {b:.2f}°")  # ~245.58°
```

### Example 5: Finding a Destination Point

Given a starting point, bearing, and distance, compute where you end up:

```python
from geodistpy import destination

# Travel 500 km due east from Berlin
lat, lon = destination((52.5200, 13.4050), 90.0, 500, metric='km')
print(f"Destination: ({lat:.4f}, {lon:.4f})")
```

### Example 6: Interpolating Waypoints Along a Geodesic

Generate evenly-spaced waypoints between two points (great for routing and visualization):

```python
from geodistpy import interpolate, midpoint

# Get the geodesic midpoint between Berlin and Paris
mid = midpoint((52.5200, 13.4050), (48.8566, 2.3522))
print(f"Midpoint: ({mid[0]:.4f}, {mid[1]:.4f})")

# Generate 4 interior waypoints along the Berlin → Paris geodesic
waypoints = interpolate((52.5200, 13.4050), (48.8566, 2.3522), n_points=4)
for i, wp in enumerate(waypoints, 1):
    print(f"  Waypoint {i}: ({wp[0]:.4f}, {wp[1]:.4f})")
```

### Example 7: Finding Points Within a Radius (Geofencing)

```python
from geodistpy import point_in_radius

# European cities
cities = [
    (48.8566, 2.3522),    # Paris
    (40.7128, -74.006),   # New York
    (51.5074, -0.1278),   # London
    (41.9028, 12.4964),   # Rome
]

# Find cities within 1500 km of Berlin
idx, dists = point_in_radius((52.5200, 13.4050), cities, 1500, metric='km')
print(f"Cities within 1500 km: indices {idx}, distances {dists.round(1)} km")
```

### Example 8: Using Different Ellipsoids

All distance, bearing, destination, interpolation, and spatial query functions accept an optional `ellipsoid` parameter. By default, WGS-84 is used. You can choose from six built-in ellipsoids or pass a custom `(a, f)` tuple:

```python
from geodistpy import geodist, ELLIPSOIDS

coord1 = (52.5200, 13.4050)  # Berlin
coord2 = (48.8566, 2.3522)   # Paris

# Use a named ellipsoid
d_grs80 = geodist(coord1, coord2, metric='km', ellipsoid='GRS-80')
print(f"GRS-80: {d_grs80:.2f} km")

# Use a custom (semi_major_axis, flattening) tuple
d_custom = geodist(coord1, coord2, metric='km', ellipsoid=(6378137.0, 1/298.257223563))
print(f"Custom: {d_custom:.2f} km")

# See all built-in ellipsoids
print(ELLIPSOIDS.keys())
# dict_keys(['WGS-84', 'GRS-80', 'Airy (1830)', 'Intl 1924', 'Clarke (1880)', 'GRS-67'])
```

Supported named ellipsoids: **WGS-84** (default), **GRS-80**, **Airy (1830)**, **Intl 1924**, **Clarke (1880)**, **GRS-67**.

### Example 9: k-Nearest Neighbours on Geodesic Distance

Find the closest points using exact ellipsoidal distances (not haversine approximation):

```python
from geodistpy import geodesic_knn

# Find the 2 nearest cities to Berlin
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

For applications that demand rapid and precise geospatial distance computations, Geodistpy is the clear choice. It offers exceptional speed improvements over Geopy, Geographiclib, and (in most tested paths) Pyproj, making it ideal for tasks involving large datasets or real-time geospatial applications. Despite its speed, Geodistpy maintains accuracy on par with Geographiclib, ensuring that fast calculations do not compromise precision.

By adopting Geodistpy, you can significantly enhance the efficiency and performance of your geospatial projects. It is a valuable tool for geospatial professionals and developers seeking both speed and accuracy in their distance computations.

To get started with Geodistpy, visit the [Geodistpy](https://github.com/pawangeek/geodistpy) and explore the documentation for comprehensive usage instructions.
