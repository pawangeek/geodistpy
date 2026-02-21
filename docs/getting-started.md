---
title: Getting Started with Geodistpy – Installation & Quick Start
description: Learn how to install geodistpy and calculate geodesic distances between coordinates in Python. Step-by-step guide with code examples.
---

# Getting Started

## Installation

You can install the `geodistpy` package using `pip`:

```bash
pip install geodistpy
```

## Quick Start Guide

The quickest way to start using the `geodistpy` package is to calculate the distance between two geographical coordinates. Here's how you can do it:

```python
from geodistpy import geodist

# Define two coordinates in (latitude, longitude) format
coord1 = (52.5200, 13.4050)  # Berlin, Germany
coord2 = (48.8566, 2.3522)   # Paris, France

# Calculate the distance between the two coordinates in kilometers
distance_km = geodist(coord1, coord2, metric='km')
print(f"Distance between Berlin and Paris: {distance_km} kilometers")
```

## Beyond Distance: Bearing, Destination, and Spatial Queries

`geodistpy` goes beyond simple distance calculations. Here are a few more things you can do right away:

```python
from geodistpy import bearing, destination, midpoint, interpolate, point_in_radius, geodesic_knn

berlin = (52.5200, 13.4050)
paris  = (48.8566, 2.3522)

# Bearing: initial direction from Berlin to Paris
print(f"Bearing: {bearing(berlin, paris):.2f}°")

# Destination: where do you end up travelling 500 km east from Berlin?
print(f"Destination: {destination(berlin, 90.0, 500, metric='km')}")

# Midpoint along the geodesic
print(f"Midpoint: {midpoint(berlin, paris)}")

# 3 evenly-spaced waypoints between Berlin and Paris
print(f"Waypoints: {interpolate(berlin, paris, n_points=3)}")

# Which of these cities are within 1000 km of Berlin?
cities = [(48.8566, 2.3522), (51.5074, -0.1278), (40.7128, -74.006)]
idx, dists = point_in_radius(berlin, cities, 1000, metric='km')
print(f"Within 1000 km: {idx}")

# 2 nearest cities to Berlin
idx, dists = geodesic_knn(berlin, cities, k=2, metric='km')
print(f"2 nearest: {idx}, distances: {dists.round(1)} km")
```

For full API details, see the [API Reference](api-reference.md).
