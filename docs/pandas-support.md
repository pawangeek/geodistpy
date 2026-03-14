---
title: Pandas & GeoPandas Support – DataFrames and After-Geocoding Workflow
description: Use geodistpy with pandas DataFrames and GeoPandas GeoDataFrames. One-to-many distances, k-NN, and point-in-radius with tabular data. Workflow after geocoding with geopy or any geocoder.
---

# Pandas & GeoPandas Support

Geodistpy integrates with **pandas** and **GeoPandas** so you can pass DataFrames (or GeoDataFrames) directly into distance and spatial-query functions. This fits the common workflow: get coordinates from a geocoder or database, then run fast geodesic distance and nearest-neighbor logic on tabular data.

## Installation (optional)

Pandas and GeoPandas are **optional** dependencies. Install them only if you use DataFrames:

```bash
pip install geodistpy[pandas]       # for pandas DataFrame support
pip install geodistpy[geopandas]    # for GeoDataFrame support (includes pandas)
```

Without these extras, all other geodistpy features work as usual; only DataFrame/GeoDataFrame arguments require the corresponding package.

---

## After geocoding: the typical workflow

Geocoding (address → latitude, longitude) is often the first step — with [geopy](https://geopy.readthedocs.io/), your own API, or a database. The next step is usually:

- **“How far from this user to these N stores?”**
- **“Which store is nearest?”**
- **“Which points are within 500 km?”**

Geodistpy is built for that: once you have coordinates, run distance and nearest-neighbor logic **fast** with geodesic accuracy.

### Example: geopy → geodistpy

```python
# 1. Get coordinates (e.g. from geopy)
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

# 3. How far from this user to each store?
from geodistpy import geodist_to_many
distances_km = geodist_to_many(user, stores, metric="km")

# 4. Which store is nearest?
from geodistpy import geodesic_knn
nearest_idx, nearest_dists = geodesic_knn(user, stores, k=1, metric="km")

# 5. Which stores are within 500 km?
from geodistpy import point_in_radius
within_idx, within_dists = point_in_radius(user, stores, 500, metric="km")
```

No projection, no haversine hacks — just fast geodesic (Vincenty) on WGS84.

---

## Using pandas DataFrames

If your points live in a **pandas DataFrame** (e.g. a table of stores or cities), you can pass the DataFrame directly. Geodistpy will use the right columns and return results aligned with the DataFrame index.

### Column names

- **Auto-detect:** Use columns named `lat`/`lon` or `latitude`/`longitude` (or `Lat`/`Lon`, `LAT`/`LON`). No extra arguments needed.
- **Custom names:** Pass `lat_col` and `lon_col` to any function that accepts a DataFrame.

### Functions that accept DataFrames

| Function | DataFrame argument | Return when DataFrame given |
|----------|--------------------|-----------------------------|
| `geodist_to_many(origin, points, ...)` | `points` | `pandas.Series` with `points.index` |
| `geodesic_knn(point, candidates, ...)` | `candidates` | `(indices, distances)` — *indices* are index labels |
| `point_in_radius(center, candidates, ...)` | `candidates` | `(indices, distances)` — *indices* are index labels |

### Example: stores in a DataFrame

```python
import pandas as pd
from geodistpy import geodist_to_many, geodesic_knn, point_in_radius

# DataFrame with lat/lon columns
stores_df = pd.DataFrame({
    "lat": [48.8566, 51.5074, 40.7128],
    "lon": [2.3522, -0.1278, -74.006],
    "name": ["Paris", "London", "New York"],
})

user = (52.52, 13.40)

# One-to-many distances — returns a Series indexed like your DataFrame
distances = geodist_to_many(user, stores_df, metric="km")

# k nearest (indices are row labels)
idx, dists = geodesic_knn(user, stores_df, k=2, metric="km")
nearest_stores = stores_df.loc[idx]

# All stores within 1000 km
idx, dists = point_in_radius(user, stores_df, 1000, metric="km")
within = stores_df.loc[idx]
```

### Custom column names

```python
df = pd.DataFrame({"y": [48.85, 51.50], "x": [2.35, -0.12]})
distances = geodist_to_many(user, df, metric="km", lat_col="y", lon_col="x")
idx, dists = geodesic_knn(user, df, k=1, metric="km", lat_col="y", lon_col="x")
```

---

## Extracting coordinates: `coordinates_from_df`

When you only need a **(lat, lon)** array from a DataFrame or GeoDataFrame (e.g. for `geodist_matrix` or custom logic), use **`coordinates_from_df`**:

```python
from geodistpy import coordinates_from_df, geodist_matrix

coords, index = coordinates_from_df(stores_df)
# coords: (n, 2) array of (latitude, longitude)
# index:  stores_df.index (for aligning results)

# Use with any function that expects an array
matrix = geodist_matrix(coords, metric="km")
```

### pandas DataFrame

- **Auto-detect:** Columns `lat`/`lon` or `latitude`/`longitude` (or `Lat`/`Lon`, `LAT`/`LON`).
- **Explicit:** `coordinates_from_df(df, lat_col="y", lon_col="x")`.
- **Returns:** `(coords, df.index)` — `coords` is shape `(n, 2)`.

### GeoPandas GeoDataFrame

- Ignores `lat_col`/`lon_col`. Uses the **geometry** column (point geometry).
- Assumes a geographic CRS (e.g. WGS84): geometry `.x` = longitude, `.y` = latitude.
- **Returns:** `(coords, gdf.index)`.

```python
import geopandas as gpd
from shapely.geometry import Point
from geodistpy import coordinates_from_df

gdf = gpd.GeoDataFrame(
    {"name": ["Paris", "London"]},
    geometry=[Point(2.35, 48.85), Point(-0.12, 51.50)],
    crs="EPSG:4326",
)
coords, index = coordinates_from_df(gdf)
# coords = [[48.85, 2.35], [51.50, -0.12]]
```

---

## One-to-many distances: `geodist_to_many`

**`geodist_to_many(origin, points, ...)`** returns the distance from a single **origin** to each of **points**. Same as `geodist_matrix([origin], points, ...)[0]` but with a clearer intent.

- **origin:** `(lat, lon)` tuple.
- **points:** Array of shape `(n, 2)` or a **pandas DataFrame / GeoDataFrame** (with `lat`/`lon` or `lat_col`/`lon_col`).
- **Returns:** `ndarray` of length *n*, or a **`pandas.Series`** (indexed by `points.index`) when *points* is a DataFrame/GeoDataFrame.

Use it whenever you have one reference point and many targets (e.g. one user, many stores).

---

## Summary

| Goal | Function | With DataFrame? |
|------|----------|-----------------|
| Distances from one point to many | `geodist_to_many(origin, points, ...)` | Yes → returns Series |
| k nearest points | `geodesic_knn(point, candidates, k=..., ...)` | Yes → indices = index labels |
| Points within radius | `point_in_radius(center, candidates, radius, ...)` | Yes → indices = index labels |
| Get (lat, lon) array from table | `coordinates_from_df(df, ...)` | Yes (DataFrame or GeoDataFrame) |

Install **`geodistpy[pandas]`** or **`geodistpy[geopandas]`** to use DataFrame/GeoDataFrame inputs. For more detail on k-NN and point-in-radius, see [Spatial Queries](spatial-queries.md). For the full API, see [API Reference](api-reference.md).
