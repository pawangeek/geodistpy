---
title: Spatial Queries – k-NN & Point-in-Radius on the WGS-84 Ellipsoid
description: Find nearest neighbours and filter points by geodesic radius using exact Vincenty distances. Geofencing, store locators, and spatial search with geodistpy.
---

# Spatial Queries: k-NN & Point-in-Radius

This guide covers two spatial query operations that use exact geodesic (Vincenty) distances on the WGS-84 ellipsoid:

- **`geodesic_knn`**: Find the k closest points to a query location
- **`point_in_radius`**: Find all points within a given distance of a centre

These fill an important gap: popular tools like scikit-learn's `BallTree` only support the haversine metric (spherical approximation), which can introduce errors of up to ~0.3% compared to true ellipsoidal distances.

## k-Nearest Neighbours (geodesic_knn)

### Why Geodesic k-NN?

Standard k-NN implementations use Euclidean distance or haversine (spherical) approximations. On the WGS-84 ellipsoid, these can produce incorrect nearest-neighbour rankings, especially for:

- Points near the poles (where longitude lines converge)
- Long-distance queries (where ellipsoidal effects are significant)
- High-precision applications (logistics, surveying, telecommunications)

Geodistpy's `geodesic_knn` uses Vincenty's inverse formula for every distance computation, ensuring **sub-millimetre accuracy** in the distance ranking.

### Basic Usage

```python
from geodistpy import geodesic_knn

# Candidate cities
cities = [
    (48.8566, 2.3522),    # 0: Paris
    (40.7128, -74.0060),  # 1: New York
    (51.5074, -0.1278),   # 2: London
    (41.9028, 12.4964),   # 3: Rome
    (59.3293, 18.0686),   # 4: Stockholm
]

# Find the 3 nearest cities to Berlin
query = (52.5200, 13.4050)
idx, dists = geodesic_knn(query, cities, k=3, metric='km')

print("3 nearest cities to Berlin:")
names = ['Paris', 'New York', 'London', 'Rome', 'Stockholm']
for i, (ci, d) in enumerate(zip(idx, dists)):
    print(f"  {i+1}. {names[ci]:>10} — {d:.1f} km")
```

### Understanding the Output

`geodesic_knn` returns a tuple of two NumPy arrays:

- **`indices`**: Shape `(k,)` — indices into the `candidates` array, ordered nearest-first
- **`distances`**: Shape `(k,)` — corresponding distances in the specified metric

```python
idx, dists = geodesic_knn(query, candidates, k=3, metric='km')

# idx[0] is the index of the nearest candidate
# dists[0] is the distance to the nearest candidate
# idx[1] is the second nearest, etc.
```

### Different Metrics

```python
from geodistpy import geodesic_knn

query = (52.5200, 13.4050)
cities = [(48.8566, 2.3522), (51.5074, -0.1278)]

# Same query, different units
_, d_m  = geodesic_knn(query, cities, k=1, metric='meter')
_, d_km = geodesic_knn(query, cities, k=1, metric='km')
_, d_mi = geodesic_knn(query, cities, k=1, metric='mile')
_, d_nm = geodesic_knn(query, cities, k=1, metric='nmi')

print(f"Nearest: {d_m[0]:.0f} m = {d_km[0]:.1f} km = {d_mi[0]:.1f} mi = {d_nm[0]:.1f} nmi")
```

### Error Handling

```python
from geodistpy import geodesic_knn

# k must be ≥ 1
geodesic_knn((0, 0), [(1, 1)], k=0)  # → ValueError

# k must not exceed number of candidates
geodesic_knn((0, 0), [(1, 1)], k=5)  # → ValueError

# Invalid coordinates
geodesic_knn((95, 0), [(1, 1)], k=1)  # → ValueError (lat > 90)
```

## Point-in-Radius

### Why Point-in-Radius?

The `point_in_radius` function finds all candidate points within a specified geodesic distance of a centre point. This is the fundamental operation behind:

- **Geofencing**: Trigger actions when devices enter/leave a zone
- **Store locators**: "Find stores within 10 km of me"
- **Proximity alerts**: Warn when approaching a restricted area
- **Service coverage**: Determine which users are within coverage radius
- **Spatial filtering**: Pre-filter candidates before detailed analysis

### Basic Usage

```python
from geodistpy import point_in_radius

# Candidate locations
locations = [
    (48.8566, 2.3522),    # 0: Paris      (~880 km from Berlin)
    (40.7128, -74.0060),  # 1: New York   (~6400 km from Berlin)
    (51.5074, -0.1278),   # 2: London     (~930 km from Berlin)
    (41.9028, 12.4964),   # 3: Rome       (~1180 km from Berlin)
    (59.3293, 18.0686),   # 4: Stockholm  (~810 km from Berlin)
]

# Find locations within 1000 km of Berlin
centre = (52.5200, 13.4050)
idx, dists = point_in_radius(centre, locations, 1000, metric='km')

names = ['Paris', 'New York', 'London', 'Rome', 'Stockholm']
print(f"Cities within 1000 km of Berlin:")
for i, (ci, d) in enumerate(zip(idx, dists)):
    print(f"  {names[ci]:>10} — {d:.1f} km")
```

### Understanding the Output

`point_in_radius` returns a tuple of two NumPy arrays:

- **`indices`**: Indices into `candidates` of points within the radius
- **`distances`**: Corresponding distances to each point within the radius

```python
idx, dists = point_in_radius(centre, candidates, radius, metric='km')

# len(idx) is the number of points found
# idx contains the original indices into candidates
# dists[i] is the distance to candidates[idx[i]]
```

### Boundary Behaviour

Points exactly on the boundary (distance = radius) are **included**:

```python
from geodistpy import point_in_radius, geodist

centre = (0.0, 0.0)
pt = (0.0, 1.0)

exact_dist = geodist(centre, pt, metric='km')
idx, _ = point_in_radius(centre, [pt], exact_dist, metric='km')
print(f"Included: {0 in idx}")  # True (boundary is inclusive)
```

### Empty Results

When no points fall within the radius, empty arrays are returned:

```python
from geodistpy import point_in_radius

idx, dists = point_in_radius((0, 0), [(48.8566, 2.3522)], 10, metric='km')
print(f"Found: {len(idx)} points")  # 0
print(f"idx: {idx}")                # []
print(f"dists: {dists}")            # []
```

## Real-World Examples

### Store Locator

```python
from geodistpy import point_in_radius

# Store coordinates (lat, lon)
stores = [
    (48.8534, 2.3488),   # Store A — central Paris
    (48.8738, 2.2950),   # Store B — near Eiffel Tower
    (48.8606, 2.3376),   # Store C — Louvre area
    (48.8156, 2.3631),   # Store D — south Paris
    (49.0097, 2.5479),   # Store E — CDG airport area
]

# User location
user = (48.8566, 2.3522)  # near Notre-Dame

# Find stores within 3 km
idx, dists = point_in_radius(user, stores, 3, metric='km')

print(f"Stores within 3 km:")
for ci, d in zip(idx, dists):
    print(f"  Store {chr(65 + ci)} — {d * 1000:.0f} m away")
```

### Geofencing Alert System

```python
from geodistpy import point_in_radius

# Restricted zones (centre points)
restricted_zones = [
    (51.4775, -0.4614),  # Heathrow Airport
    (51.1537, -0.1821),  # Gatwick Airport
]
zone_radius_km = 5  # 5 km exclusion zone

# Check if a drone position violates any zone
drone_pos = (51.4700, -0.4500)

for i, zone in enumerate(restricted_zones):
    idx, dists = point_in_radius(zone, [drone_pos], zone_radius_km, metric='km')
    if len(idx) > 0:
        print(f"⚠️  ALERT: Within {dists[0]:.2f} km of restricted zone {i}")
    else:
        print(f"✅ Clear of restricted zone {i}")
```

### Combining k-NN with Point-in-Radius

A common pattern: first filter by radius, then rank the results:

```python
from geodistpy import point_in_radius, geodesic_knn
import numpy as np

# Many candidate locations
candidates = [
    (48.8566, 2.3522),    # Paris
    (40.7128, -74.0060),  # New York
    (51.5074, -0.1278),   # London
    (41.9028, 12.4964),   # Rome
    (59.3293, 18.0686),   # Stockholm
    (50.0755, 14.4378),   # Prague
    (52.2297, 21.0122),   # Warsaw
    (47.4979, 19.0402),   # Budapest
]

query = (52.5200, 13.4050)  # Berlin

# Step 1: Filter to within 1500 km
idx_in_range, _ = point_in_radius(query, candidates, 1500, metric='km')
nearby = [candidates[i] for i in idx_in_range]

# Step 2: Find the 3 nearest among those
if len(nearby) >= 3:
    knn_idx, knn_dists = geodesic_knn(query, nearby, k=3, metric='km')
    print("Top 3 nearest within 1500 km:")
    for ki, kd in zip(knn_idx, knn_dists):
        original_idx = idx_in_range[ki]
        print(f"  Index {original_idx}: {kd:.1f} km")
```

## Haversine vs Vincenty: Why It Matters

| Aspect | Haversine (spherical) | Vincenty (ellipsoidal) |
|--------|-----------------------|------------------------|
| Earth model | Perfect sphere (R=6371 km) | WGS-84 ellipsoid |
| Mean error | ~0.3% (~3 km per 1000 km) | ~0.000001% (~9 µm) |
| Max error | ~0.5% at equator | Sub-millimetre |
| Ranking errors | Possible for close candidates | None |
| Speed (geodistpy) | ~0.3 µs/call | ~0.5 µs/call |

For most k-NN and radius queries, the **ranking** is what matters. In cases where two candidates are close in distance, haversine's ~0.3% error can swap their order, returning the wrong "nearest" point. Vincenty eliminates this risk entirely.

## Performance Considerations

| Operation | 10 candidates | 100 candidates | 1,000 candidates |
|-----------|--------------|----------------|-------------------|
| `geodesic_knn` | ~5 µs | ~50 µs | ~500 µs |
| `point_in_radius` | ~5 µs | ~50 µs | ~500 µs |

Both functions compute distances sequentially (one Vincenty call per candidate). For very large candidate sets (>100,000 points), consider:

1. **Pre-filtering** with a bounding box on lat/lon before calling these functions
2. Using `geodist_matrix` for batch computation when you need all pairwise distances

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| k = n (all candidates) | Returns all candidates sorted by distance |
| Coincident query and candidate | Distance = 0, included in results |
| Near-antipodal points | GeographicLib fallback ensures correct distance |
| Points at the poles | Handled correctly |
| Empty radius result | Returns empty arrays `(array([]), array([]))` |
