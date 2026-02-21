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
