---
title: Geodesic Interpolation & Midpoints – Waypoints Along the WGS-84 Ellipsoid
description: Generate evenly-spaced waypoints, midpoints, and geodesic paths between coordinates using Vincenty's formulas with geodistpy.
---

# Geodesic Interpolation & Midpoints

This guide covers generating points along a geodesic path — essential for route visualization, path rendering on maps, and spatial analysis.

## Why Geodesic Interpolation?

When you draw a "straight line" between two points on a map, the actual shortest path on the Earth (the geodesic) curves. Simply interpolating latitude and longitude linearly produces incorrect paths, especially over long distances. Geodesic interpolation ensures waypoints lie on the true shortest path across the WGS-84 ellipsoid.

**Common use cases:**

- **Route visualization**: Draw smooth geodesic arcs on maps
- **Flight path rendering**: Accurate great-circle routes for aviation
- **Cable/pipeline routing**: Plan paths along the Earth's surface
- **Animation**: Smooth movement between geographic coordinates
- **Spatial sampling**: Sample points at regular intervals along a path

## Midpoint

The `midpoint()` function returns the geographic point exactly halfway along the geodesic between two endpoints.

### Basic Usage

```python
from geodistpy import midpoint, geodist

# Midpoint between two equatorial points
mid = midpoint((0.0, 0.0), (0.0, 10.0))
print(f"Midpoint: ({mid[0]:.4f}, {mid[1]:.4f})")
# → (0.0000, 5.0000)

# Midpoint between Berlin and Paris
mid = midpoint((52.5200, 13.4050), (48.8566, 2.3522))
print(f"Midpoint: ({mid[0]:.4f}, {mid[1]:.4f})")
```

### Verifying the Midpoint

The midpoint should be equidistant from both endpoints:

```python
from geodistpy import midpoint, geodist

A = (52.5200, 13.4050)  # Berlin
B = (48.8566, 2.3522)   # Paris

mid = midpoint(A, B)

d_A = geodist(A, mid, metric='km')
d_B = geodist(mid, B, metric='km')
d_total = geodist(A, B, metric='km')

print(f"A → mid:    {d_A:.2f} km")
print(f"mid → B:    {d_B:.2f} km")
print(f"A → B:      {d_total:.2f} km")
print(f"Sum:        {d_A + d_B:.2f} km")
# d_A ≈ d_B ≈ d_total / 2
```

### Midpoint is Symmetric

```python
from geodistpy import midpoint

A = (52.5200, 13.4050)
B = (48.8566, 2.3522)

m1 = midpoint(A, B)
m2 = midpoint(B, A)

print(f"midpoint(A,B) = ({m1[0]:.6f}, {m1[1]:.6f})")
print(f"midpoint(B,A) = ({m2[0]:.6f}, {m2[1]:.6f})")
# These are identical
```

## Interpolation (Multiple Waypoints)

The `interpolate()` function generates N evenly-spaced interior waypoints that divide a geodesic into N+1 equal segments. The endpoints are **not** included in the output.

### Basic Usage

```python
from geodistpy import interpolate

# Single midpoint (same as midpoint())
pts = interpolate((0.0, 0.0), (0.0, 10.0), n_points=1)
print(pts)  # [(0.0, 5.0)]

# Four interior waypoints → 5 equal segments
pts = interpolate((0.0, 0.0), (0.0, 10.0), n_points=4)
for i, p in enumerate(pts, 1):
    print(f"  Waypoint {i}: ({p[0]:.4f}, {p[1]:.4f})")
# → ~2°, ~4°, ~6°, ~8° longitude
```

### Understanding the Output

For `n_points=N`, the geodesic is divided into N+1 segments:

```
A ----●----●----●----●---- B     (n_points=4)
     wp1  wp2  wp3  wp4

Segments:  A→wp1, wp1→wp2, wp2→wp3, wp3→wp4, wp4→B
           (all approximately equal length)
```

The endpoints A and B are **not** in the returned list. To get the complete path including endpoints:

```python
from geodistpy import interpolate

A = (52.5200, 13.4050)
B = (48.8566, 2.3522)

waypoints = interpolate(A, B, n_points=4)
full_path = [A] + waypoints + [B]

for i, p in enumerate(full_path):
    print(f"  Point {i}: ({p[0]:.4f}, {p[1]:.4f})")
```

### Verifying Equal Spacing

```python
from geodistpy import interpolate, geodist

A = (52.5200, 13.4050)  # Berlin
B = (48.8566, 2.3522)   # Paris

waypoints = interpolate(A, B, n_points=4)
full_path = [A] + waypoints + [B]

total = geodist(A, B, metric='km')
print(f"Total distance: {total:.2f} km")
print(f"Expected segment: {total / 5:.2f} km\n")

for i in range(len(full_path) - 1):
    d = geodist(full_path[i], full_path[i + 1], metric='km')
    print(f"  Segment {i}→{i+1}: {d:.2f} km")
```

## Real-World Examples

### Flight Path Rendering

Generate a smooth geodesic arc for display on a map:

```python
from geodistpy import interpolate

# New York → London flight path
nyc = (40.7128, -74.0060)
london = (51.5074, -0.1278)

# 50 waypoints for a smooth curve
path = interpolate(nyc, london, n_points=50)
full_path = [nyc] + path + [london]

# full_path now contains 52 (lat, lon) tuples
# ready for plotting on a map library like Folium, Plotly, or Matplotlib
print(f"Path has {len(full_path)} points")
for p in full_path[:5]:
    print(f"  ({p[0]:.4f}, {p[1]:.4f})")
print(f"  ...")
```

### Sampling at Fixed Intervals

Instead of a fixed number of points, you can compute the number of points for a desired spacing:

```python
from geodistpy import interpolate, geodist

A = (40.7128, -74.0060)  # New York
B = (51.5074, -0.1278)   # London

total_km = geodist(A, B, metric='km')
desired_spacing_km = 100  # one point every 100 km

n = max(1, int(total_km / desired_spacing_km) - 1)
waypoints = interpolate(A, B, n_points=n)

print(f"Total distance: {total_km:.0f} km")
print(f"Generated {n} waypoints (~{total_km / (n + 1):.0f} km apart)")
```

### Chaining Multiple Segments

For a multi-leg route, interpolate each segment separately:

```python
from geodistpy import interpolate

# Multi-city route: Berlin → Paris → London
legs = [
    ((52.5200, 13.4050), (48.8566, 2.3522)),   # Berlin → Paris
    ((48.8566, 2.3522), (51.5074, -0.1278)),    # Paris → London
]

full_route = []
for start, end in legs:
    if full_route:
        full_route.pop()  # avoid duplicate at junction
    segment = [start] + interpolate(start, end, n_points=10) + [end]
    full_route.extend(segment)

print(f"Route has {len(full_route)} points across {len(legs)} legs")
```

## How It Works

The interpolation algorithm uses two steps:

1. **Vincenty inverse**: Compute the total geodesic distance `s` and forward azimuth `α₁` from point1 to point2.
2. **Vincenty direct** (repeated): For each waypoint `i` in `[1, ..., N]`, compute the point at distance `s × i / (N+1)` from point1 along azimuth `α₁`.

This approach produces points that lie exactly on the geodesic between the two endpoints, not on a great-circle approximation.

!!! note "Why not spherical interpolation?"
    Spherical (SLERP) interpolation assumes a perfect sphere and can accumulate errors of up to ~21 km over intercontinental distances. Geodistpy's ellipsoidal interpolation uses the WGS-84 ellipsoid for maximum accuracy.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Coincident points | Returns N copies of the input point |
| Very short distance | Works correctly down to millimeter scale |
| Near-antipodal points | Falls back to GeographicLib for convergence |
| Points on the equator | Longitude interpolation is nearly linear |
| Pole-crossing paths | Handled correctly by Vincenty direct |
