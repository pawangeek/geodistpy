---
title: API Reference – Geodistpy Functions & Parameters
description: Complete API reference for geodistpy. Documentation for geodist, geodist_matrix, greatcircle, and greatcircle_matrix functions with parameters, return values, and examples.
---

# API Reference

This page provides the complete API reference for the `geodistpy` package. All coordinates are expected in **(latitude, longitude)** format using the WGS 84 coordinate system.

## Distance Functions

### `geodist`

```python
geodist(coords1, coords2, metric="meter")
```

Calculate the geodesic distance between two coordinates or two lists of coordinates using **Vincenty's inverse formula** (sub-millimeter accuracy).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `coords1` | `tuple` or `array-like` | First coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `coords2` | `tuple` or `array-like` | Second coordinate(s) as `(lat, lon)` or array of shape `(n, 2)` |
| `metric` | `str` | Unit of measurement: `'meter'`, `'km'`, `'mile'`, or `'nmi'`. Default: `'meter'` |

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
geodist_matrix(coords1, coords2=None, metric="meter")
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

## Supported Metrics

All distance functions accept a `metric` parameter with one of the following values:

| Value | Unit | Conversion |
|---|---|---|
| `'meter'` | Meters | Base unit |
| `'km'` | Kilometers | ÷ 1,000 |
| `'mile'` | Statute miles | ÷ 1,609.344 |
| `'nmi'` | Nautical miles | ÷ 1,852 |
