---
title: API Reference – Geodistpy Functions & Parameters
description: Complete API reference for geodistpy. Documentation for geodist, geodist_matrix, greatcircle, and greatcircle_matrix functions with parameters, return values, and examples.
---

# API Reference

This page provides the complete API reference for the `geodistpy` package. All coordinates are expected in **(latitude, longitude)** format using the WGS 84 coordinate system.

## Distance Functions

### `geodist`

```python
geodist(coords1, coords2, metric="meter", ellipsoid="WGS-84")
```

Calculate the geodesic distance between two coordinates or two lists of coordinates using **Vincenty's inverse formula** (sub-millimeter accuracy).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `coords1` | `tuple` or `array-like` | First coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `coords2` | `tuple` or `array-like` | Second coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `metric` | `str` | Unit of measurement: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` (e.g. `'WGS-84'`, `'GRS-80'`) or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `float` or `ndarray` — Distance(s) in the specified unit.

**Raises:**

- `ValueError` — If coordinates don't have expected shape, or lat/lon values are out of range.

**Examples:**

```python
from geodistpy import geodist

# Single pair
>>> geodist((52.5200, 13.4050), (48.8566, 2.3522), metric='km')
878.389841013836

# Multiple pairs
>>> coords1 = [(37.7749, -122.4194), (34.0522, -118.2437)]
>>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
>>> geodist(coords1, coords2, metric='mile')
array([2449.92107243, 1745.82567572])

# Same point returns zero
>>> geodist((37.7749, -122.4194), (37.7749, -122.4194))
0.0
```

---

### `greatcircle`

```python
greatcircle(coords1, coords2, metric="meter")
```

Calculate the distance between two coordinates using the **Great Circle approximation** with Andoyer-Lambert flattening correction (~19 m mean accuracy).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `coords1` | `tuple` or `array-like` | First coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `coords2` | `tuple` or `array-like` | Second coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `metric` | `str` | Unit of measurement: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |

**Returns:** `float` or `ndarray` — Distance(s) in the specified unit.

**Raises:**

- `ValueError` — If coordinates don't have expected shape, or lat/lon values are out of range.

!!! note
    The Great Circle formula with Andoyer-Lambert correction assumes an oblate spheroid (WGS84 flattening). It is faster than Vincenty but less accurate (~19 m mean error vs ~9 µm).

---

## Matrix Functions

### `geodist_matrix`

```python
geodist_matrix(coords1, coords2=None, metric="meter", ellipsoid="WGS-84")
```

Compute a **pairwise distance matrix** between all coordinate combinations using Vincenty's inverse formula with Numba-parallel execution.

- If only `coords1` is given: computes distances between all pairs in `coords1` → `dist[i, j] = distance(X[i], X[j])`
- If `coords2` is also given: computes cross-distances → `dist[i, j] = distance(XA[i], XB[j])`

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `coords1` | `list of tuples` or `array-like` | Coordinates as `[(lat, lon), ...]` or array of shape `(n, 2)` |
| `coords2` | `list of tuples` or `None` | Optional second set of coordinates. Default: `None` |
| `metric` | `str` | Unit of measurement: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `ndarray` — Distance matrix.

**Raises:**

- `ValueError` — If coordinates don't have expected shape, or lat/lon values are out of range.

**Examples:**

```python
from geodistpy import geodist_matrix

# Self-distance matrix
>>> coords = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
>>> geodist_matrix(coords, metric='km')
array([[   0.        ,  878.38984101, 8786.58652276],
       [ 878.38984101,    0.        , 9525.03650888],
       [8786.58652276, 9525.03650888,    0.        ]])

# Cross-distance matrix
>>> coords2 = [(40.7128, -74.0060), (41.8781, -87.6298)]
>>> geodist_matrix(coords, coords2, metric='mile')
array([[ 3060.81391478, 2437.78157493],
       [ 4290.62813902, 1745.82567572],
       [ 2449.92107243, 1746.57308007]])
```

---

### `greatcircle_matrix`

```python
greatcircle_matrix(coords1, coords2=None, metric="meter")
```

Compute a **pairwise distance matrix** using the Great Circle approximation with Andoyer-Lambert correction and Numba-parallel execution.

- If only `coords1` is given: computes distances between all pairs in `coords1`
- If `coords2` is also given: computes cross-distances

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `coords1` | `list of tuples` or `array-like` | Coordinates as `[(lat, lon), ...]` or array of shape `(n, 2)` |
| `coords2` | `list of tuples` or `None` | Optional second set of coordinates. Default: `None` |
| `metric` | `str` | Unit of measurement: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |

**Returns:** `ndarray` — Distance matrix.

**Raises:**

- `ValueError` — If coordinates don't have expected shape, or lat/lon values are out of range.

**Examples:**

```python
from geodistpy import greatcircle_matrix

# Self-distance matrix
>>> coords = [(52.5200, 13.4050), (48.8566, 2.3522), (37.7749, -122.4194)]
>>> greatcircle_matrix(coords, metric='km')
array([[   0.        ,  878.38984101, 8786.58652276],
       [ 878.38984101,    0.        , 9525.03650888],
       [8786.58652276, 9525.03650888,    0.        ]])
```

---

## Spatial Query Functions

### `bearing`

```python
bearing(point1, point2, ellipsoid="WGS-84")
```

Compute the **initial bearing** (forward azimuth) from *point1* to *point2* on an ellipsoid using Vincenty's inverse formula. The bearing is measured clockwise from true north and returned in the range **[0, 360)** degrees.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `point1` | `tuple` | Starting point as `(lat, lon)` in degrees |
| `point2` | `tuple` | Destination point as `(lat, lon)` in degrees |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `float` — Forward azimuth in degrees (0–360).

**Examples:**

```python
from geodistpy import bearing

>>> bearing((52.5200, 13.4050), (48.8566, 2.3522))   # Berlin → Paris
245.58...

>>> bearing((0.0, 0.0), (0.0, 1.0))                  # Due east on the equator
90.0
```

---

### `destination`

```python
destination(point, bearing_deg, distance, metric="meter", ellipsoid="WGS-84")
```

Compute the **destination point** given a starting point, initial bearing, and distance along the geodesic on an ellipsoid (Vincenty direct formula).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `point` | `tuple` | Starting point as `(lat, lon)` in degrees |
| `bearing_deg` | `float` | Initial bearing in degrees clockwise from north |
| `distance` | `float` | Distance to travel in the unit specified by `metric` |
| `metric` | `str` | Unit for *distance*: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `tuple` — Destination point as `(latitude, longitude)` in degrees.

**Examples:**

```python
from geodistpy import destination

>>> destination((52.5200, 13.4050), 245.0, 879.0, metric='km')
(48.85..., 2.35...)

>>> destination((0.0, 0.0), 90.0, 111.32, metric='km')
(0.0, 1.0...)
```

---

### `interpolate`

```python
interpolate(point1, point2, n_points=1, ellipsoid="WGS-84")
```

Return **evenly-spaced waypoints** along the geodesic from *point1* to *point2* on an ellipsoid. When `n_points=1` the function returns the **midpoint**. For `n_points=N` it returns *N* interior points that divide the geodesic into *N + 1* equal-length segments (endpoints are **not** included).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `point1` | `tuple` | Start point as `(lat, lon)` in degrees |
| `point2` | `tuple` | End point as `(lat, lon)` in degrees |
| `n_points` | `int` | Number of interior waypoints to return. Default: `1` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `list of tuples` — Waypoints as `[(lat, lon), ...]`, ordered from *point1* towards *point2*.

**Examples:**

```python
from geodistpy import interpolate

>>> interpolate((0.0, 0.0), (0.0, 10.0), n_points=1)
[(0.0, 5.0...)]

>>> interpolate((0.0, 0.0), (0.0, 10.0), n_points=4)
[(0.0, 2.0...), (0.0, 4.0...), (0.0, 6.0...), (0.0, 8.0...)]
```

---

### `midpoint`

```python
midpoint(point1, point2, ellipsoid="WGS-84")
```

Return the **geodesic midpoint** between two points on an ellipsoid. Convenience wrapper around `interpolate(point1, point2, n_points=1)`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `point1` | `tuple` | First point as `(lat, lon)` in degrees |
| `point2` | `tuple` | Second point as `(lat, lon)` in degrees |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `tuple` — Midpoint as `(latitude, longitude)` in degrees.

**Examples:**

```python
from geodistpy import midpoint

>>> midpoint((0.0, 0.0), (0.0, 10.0))
(0.0, 5.0...)
```

---

### `point_in_radius`

```python
point_in_radius(center, candidates, radius, metric="meter", ellipsoid="WGS-84")
```

Find all candidate points that lie **within a given geodesic radius** of a center point on an ellipsoid. Useful for geofencing, store-locator queries, and spatial filtering.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `center` | `tuple` | Reference point as `(lat, lon)` in degrees |
| `candidates` | `array-like` | Array of candidate points `[(lat, lon), ...]` with shape `(n, 2)` |
| `radius` | `float` | Radius threshold in the unit specified by `metric` |
| `metric` | `str` | Unit for *radius*: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `tuple (indices, distances)` — *indices* is an ndarray of int (indices of points within the radius); *distances* is an ndarray of float (corresponding distances).

**Examples:**

```python
from geodistpy import point_in_radius

>>> pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
>>> idx, dists = point_in_radius((52.5200, 13.4050), pts, 1000, metric='km')
>>> idx
array([0, 2])    # Paris and London are within 1000 km of Berlin
```

---

### `geodesic_knn`

```python
geodesic_knn(point, candidates, k=1, metric="meter", ellipsoid="WGS-84")
```

Find the **k nearest neighbours** to a query point among candidates using exact geodesic (Vincenty) distances on an ellipsoid. This fills the gap left by `sklearn.neighbors.BallTree` which only supports the haversine (spherical) metric.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `point` | `tuple` | Query point as `(lat, lon)` in degrees |
| `candidates` | `array-like` | Array of candidate points `[(lat, lon), ...]` with shape `(n, 2)` |
| `k` | `int` | Number of nearest neighbours to return. Default: `1` |
| `metric` | `str` | Unit for returned distances: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |
| `ellipsoid` | `str` or `tuple` | Ellipsoid model: a named key from `ELLIPSOIDS` or a custom `(a, f)` tuple. Default: `'WGS-84'` |

**Returns:** `tuple (indices, distances)` — *indices* is an ndarray of int, shape `(k,)` (indices of the *k* closest points, ordered nearest-first); *distances* is an ndarray of float, shape `(k,)`.

**Raises:**

- `ValueError` — If `k < 1`, `k > n`, or coordinates are out of range.

**Examples:**

```python
from geodistpy import geodesic_knn

>>> pts = [(48.8566, 2.3522), (40.7128, -74.006), (51.5074, -0.1278)]
>>> idx, dists = geodesic_knn((52.5200, 13.4050), pts, k=2, metric='km')
>>> idx
array([0, 2])    # Paris (~880 km) and London (~930 km) are nearest
```

---

## Supported Metrics

All distance and spatial query functions accept a `metric` parameter with one of the following values:

| Value | Unit | Conversion |
|---|---|---|
| `'meter'` | Meters | Base unit |
| `'km'` | Kilometers | ÷ 1,000 |
| `'mile'` | Statute miles | ÷ 1,609.344 |
| `'nmi'` | Nautical miles | ÷ 1,852 |

---

## Supported Ellipsoids

All Vincenty-based functions (`geodist`, `geodist_matrix`, `bearing`, `destination`, `interpolate`, `midpoint`, `point_in_radius`, `geodesic_knn`) accept an optional `ellipsoid` parameter. You can pass either:

- A **named string** from the built-in `ELLIPSOIDS` dictionary
- A **custom tuple** `(semi_major_axis, flattening)` where *a* is in meters

### Built-in Ellipsoids (`ELLIPSOIDS`)

```python
from geodistpy import ELLIPSOIDS
print(ELLIPSOIDS)
```

| Name | Semi-major axis *a* (m) | Flattening *f* | Use Case |
|---|---|---|---|
| `'WGS-84'` | 6,378,137.0 | 1 / 298.257223563 | GPS, global mapping (default) |
| `'GRS-80'` | 6,378,137.0 | 1 / 298.257222101 | Geodetic reference, ITRF |
| `'Airy (1830)'` | 6,377,563.396 | 1 / 299.3249646 | Ordnance Survey Great Britain |
| `'Intl 1924'` | 6,378,388.0 | 1 / 297.0 | European Datum 1950 |
| `'Clarke (1880)'` | 6,378,249.145 | 1 / 293.465 | Africa, France (historic) |
| `'GRS-67'` | 6,378,160.0 | 1 / 298.25 | South American Datum 1969 |

### Examples

```python
from geodistpy import geodist, bearing, destination, ELLIPSOIDS

berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

# Named ellipsoid
d = geodist(berlin, paris, metric='km', ellipsoid='GRS-80')
b = bearing(berlin, paris, ellipsoid='Airy (1830)')

# Custom ellipsoid as (a, f) tuple
d = geodist(berlin, paris, ellipsoid=(6378137.0, 1/298.257223563))
lat, lon = destination(berlin, 90.0, 500, metric='km', ellipsoid=(6378388.0, 1/297.0))
```
