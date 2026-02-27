---
title: Bearing & Destination – Vincenty Direct/Inverse on the WGS-84 Ellipsoid
description: Learn how to compute bearings (forward azimuth) and destination points using Vincenty's direct and inverse formulas on the WGS-84 ellipsoid with geodistpy.
---

# Bearing & Destination

This guide covers two complementary operations on the WGS-84 ellipsoid:

- **Bearing** (inverse problem): Given two points, what is the initial direction from one to the other?
- **Destination** (direct problem): Given a starting point, a direction, and a distance, where do you end up?

Together, these complete Vincenty's inverse/direct pair and enable powerful geospatial workflows.

## Understanding Bearing

A **bearing** (or **forward azimuth**) is the initial compass direction you would travel from point A to reach point B along the shortest path on the Earth's surface (the geodesic). It is measured in degrees clockwise from true north:

| Bearing | Direction |
|---------|-----------|
| 0° | North |
| 90° | East |
| 180° | South |
| 270° | West |

!!! note "Bearing changes along a geodesic"
    On an ellipsoid, the bearing is **not constant** along a geodesic (unlike a rhumb line). The bearing at point A towards B differs from the bearing at B towards A by roughly 180° — but not exactly, due to the Earth's curvature and flattening. This is why Vincenty's inverse formula computes both a *forward azimuth* and a *back azimuth*.

### Basic Usage

```python
from geodistpy import bearing

# Berlin → Paris
b = bearing((52.5200, 13.4050), (48.8566, 2.3522))
print(f"Bearing: {b:.2f}°")  # ~245.58° (roughly southwest)

# Cardinal directions on the equator
print(bearing((0, 0), (0, 1)))    # 90.0  (due east)
print(bearing((0, 0), (1, 0)))    # 0.0   (due north)
print(bearing((1, 0), (0, 0)))    # 180.0 (due south)
print(bearing((0, 1), (0, 0)))    # 270.0 (due west)
```

### Bearing is Not Symmetric

The bearing from A → B is different from B → A:

```python
from geodistpy import bearing

berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

b_ab = bearing(berlin, paris)
b_ba = bearing(paris, berlin)

print(f"Berlin → Paris: {b_ab:.2f}°")   # ~245.58°
print(f"Paris → Berlin: {b_ba:.2f}°")   # ~58.29°
print(f"Difference: {abs(b_ab - b_ba):.2f}°")  # ~187.29° (not exactly 180°)
```

This difference from 180° arises because the geodesic curves on the ellipsoid.

### Use Cases

- **Navigation**: Determine which direction to travel
- **Compass headings**: Convert geodesic paths to bearing instructions
- **Sector analysis**: Classify the direction of movement (N/S/E/W quadrant)
- **Antenna alignment**: Point directional antennas towards a target

## Understanding Destination

The **destination** function solves Vincenty's direct problem: given a starting point, an initial bearing, and a distance, it computes the endpoint.

### Basic Usage

```python
from geodistpy import destination

# Travel 500 km due east from Berlin
lat, lon = destination((52.5200, 13.4050), 90.0, 500, metric='km')
print(f"Destination: ({lat:.4f}, {lon:.4f})")

# Travel 100 miles due north from the equator
lat, lon = destination((0.0, 0.0), 0.0, 100, metric='mile')
print(f"Destination: ({lat:.4f}, {lon:.4f})")
```

### Supported Distance Units

The `metric` parameter controls the unit of the `distance` argument:

```python
from geodistpy import destination

start = (52.5200, 13.4050)

# All produce the same destination:
destination(start, 90.0, 500_000, metric='meter')
destination(start, 90.0, 500, metric='km')
destination(start, 90.0, 500 / 1.609344, metric='mile')
destination(start, 90.0, 500 / 1.852, metric='nmi')
```

### Longitude Normalisation

Destination longitudes are automatically normalised to the range [-180, 180]:

```python
from geodistpy import destination

# Travel east past the antimeridian
lat, lon = destination((0.0, 170.0), 90.0, 2000, metric='km')
print(f"lon = {lon:.2f}")  # Normalised to [-180, 180]
```

## Roundtrip: Bearing → Destination

The `bearing` and `destination` functions are inverses of each other. You can verify this with a roundtrip:

```python
from geodistpy import bearing, destination, geodist

# Given two points
A = (52.5200, 13.4050)  # Berlin
B = (48.8566, 2.3522)   # Paris

# Step 1: Compute bearing and distance
b = bearing(A, B)
d = geodist(A, B, metric='km')
print(f"Bearing: {b:.4f}°, Distance: {d:.2f} km")

# Step 2: Reconstruct B using destination
B_reconstructed = destination(A, b, d, metric='km')
print(f"Original:      ({B[0]:.4f}, {B[1]:.4f})")
print(f"Reconstructed: ({B_reconstructed[0]:.4f}, {B_reconstructed[1]:.4f})")
# The two should match to ~4 decimal places
```

## Real-World Example: Search Sector

Create a search area by projecting points at various bearings from a centre:

```python
from geodistpy import destination

# Centre of search area
centre = (48.8566, 2.3522)  # Paris
radius_km = 50

# Generate 8 compass points around the centre
compass = {
    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
}

for name, bearing_deg in compass.items():
    lat, lon = destination(centre, bearing_deg, radius_km, metric='km')
    print(f"  {name:>2}: ({lat:.4f}, {lon:.4f})")
```

## How It Works: Vincenty's Formulas

### Inverse (Bearing)

Vincenty's inverse formula iteratively solves for the geodesic distance and azimuths between two points on an ellipsoid. The key outputs are:

- **s** — geodesic distance
- **α₁** — forward azimuth (bearing at start point)
- **α₂** — back azimuth (bearing at end point)

Geodistpy's `geodesic_vincenty_inverse_full()` returns all three in a single Numba-JIT pass. The high-level `bearing()` function extracts just the forward azimuth.

### Direct (Destination)

Vincenty's direct formula takes a starting point, forward azimuth α₁, and distance s, then iteratively computes:

- **φ₂, λ₂** — latitude and longitude of the destination point
- **α₂** — back azimuth at the destination

Geodistpy's `geodesic_vincenty_direct()` implements this as a Numba-JIT function for maximum performance.

### Convergence and Fallback

Both formulas iterate until convergence (threshold: 10⁻¹¹). In the rare case (~0.01%) that the inverse formula fails to converge (typically near-antipodal points), geodistpy falls back to the slower but always-convergent GeographicLib algorithm.

## Performance

Both `bearing()` and `destination()` benefit from Numba JIT compilation:

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| `bearing()` | ~0.5 µs | Same cost as distance computation |
| `destination()` | ~0.5 µs | Vincenty direct converges quickly |

These are fast enough for real-time applications with thousands of calls per second.

## Using Different Ellipsoids

Both `bearing()` and `destination()` accept an optional `ellipsoid` parameter. By default, WGS-84 is used. You can choose from six built-in ellipsoids or pass a custom `(a, f)` tuple:

```python
from geodistpy import bearing, destination, ELLIPSOIDS

# Bearing on the GRS-80 ellipsoid
b = bearing((52.5200, 13.4050), (48.8566, 2.3522), ellipsoid='GRS-80')
print(f"GRS-80 bearing: {b:.4f}°")

# Destination on the Airy 1830 ellipsoid
lat, lon = destination((52.5200, 13.4050), 90.0, 500, metric='km', ellipsoid='Airy (1830)')
print(f"Airy destination: ({lat:.4f}, {lon:.4f})")

# Custom ellipsoid as (semi_major_axis, flattening) tuple
lat, lon = destination((0, 0), 45.0, 1000, metric='km', ellipsoid=(6378388.0, 1/297.0))
print(f"Custom destination: ({lat:.4f}, {lon:.4f})")

# Available named ellipsoids
print(ELLIPSOIDS.keys())
# dict_keys(['WGS-84', 'GRS-80', 'Airy (1830)', 'Intl 1924', 'Clarke (1880)', 'GRS-67'])
```

Supported named ellipsoids: **WGS-84** (default), **GRS-80**, **Airy (1830)**, **Intl 1924**, **Clarke (1880)**, **GRS-67**.
