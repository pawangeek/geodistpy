# Geodistpy: Fast and Accurate Geospatial Distance Computations

[![pypi](https://img.shields.io/pypi/v/geodistpy?label=PyPI&logo=PyPI&logoColor=white&color=blue)](https://pypi.python.org/pypi/geodistpy)
[![lint](https://github.com/pawangeek/geodistpy/actions/workflows/lint.yml/badge.svg)](https://github.com/pawangeek/geodistpy/actions/workflows/lint.yml)
[![Build status](https://ci.appveyor.com/api/projects/status/iqux1vla5rm8bi8r?svg=true)](https://ci.appveyor.com/project/pawangeek/geodistpy)
[![Github Build](https://github.com/pawangeek/geodistpy/actions/workflows/publish_github.yml/badge.svg)](https://github.com/pawangeek/geodistpy/actions/workflows/publish_github.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geodistpy?label=Python&logo=Python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Geodistpy is a powerful Python library designed for lightning-fast geospatial distance computations. In this README, we'll compare Geodistpy with two other popular libraries, Geopy and Geographiclib, to highlight the significant performance advantages of Geodistpy.

* Documentation: https://pawangeek.github.io/geodistpy/
* Github Repo: https://github.com/pawangeek/geodistpy
* PyPI: https://pypi.org/project/geodistpy/

## Speed Comparison

```python
# Import libraries
from geopy.distance import geodesic as geodesic_geopy
from geographiclib.geodesic import Geodesic as geodesic_gglib
from geokernels.geodesics import geodesic_vincenty

# Define two coordinates
coord1 = (52.5200, 13.4050)  # Berlin
coord2 = (48.8566, 2.3522)   # Paris

# Calculate distance with Geopy (based on Geographiclib)
distance_geopy = geodesic_geopy(coord1, coord2).meters

# Calculate distance with Geographiclib
distance_gglib = geodesic_gglib.WGS84.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])['s12']

# Calculate distance with Geokernels
distance_geokernels = geodesic_vincenty(coord1, coord2)

# Print the results
print(f"Distance between Berlin and Paris:")
print(f"Geopy: {distance_geopy} meters")
print(f"Geographiclib: {distance_gglib} meters")
print(f"Geokernels: {distance_geokernels} meters")
```

We conducted a speed comparison between Geodistpy, Geopy, and Geographiclib using 1000 random samples of coordinates (latitude and longitude). The goal was to calculate all pairwise distances between these coordinates.

### Geopy (Geodesic from Geographiclib)

- Computation Time: Approximately 53.356 seconds
- Accuracy: Comparable to Geographiclib
- Geopy is widely known but relatively slow for distance calculations.

### Geographiclib

- Computation Time: Approximately 36.824 seconds
- Accuracy: High
- Geographiclib is established but still lags in terms of speed.

### Geodistpy (Accelerated Vincenty's Inverse)

- Computation Time: Approximately 0.701 seconds (initial run, including Numba compilation) and 0.393 seconds (subsequent runs)
- Accuracy: High, comparable to Geographiclib
- Geodistpy uses an optimized Vincenty's Inverse method for blazingly fast distance calculations.

## Performance Comparison

- Geodistpy is 78 to 142 times faster than Geopy.
- Geodistpy is 53 to 94 times faster than Geographiclib.

## Context and Background

The Python package `geodistpy` is a versatile library designed for geospatial calculations involving distances between geographical coordinates. It is built on the principles of geodesy and uses the WGS 84 coordinate system, which is commonly used in GPS and mapping applications.

## Why it was Created

The package was created to simplify and standardize geospatial distance calculations. Geographical distance calculations can be complex due to the curvature of the Earth's surface, and this library abstracts away those complexities, allowing users to focus on their specific geospatial tasks.

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

## Conclusion

For applications that demand rapid and precise geospatial distance computations, Geodistpy is the clear choice. It offers exceptional speed improvements over both Geopy and Geographiclib, making it ideal for tasks involving large datasets or real-time geospatial applications. Despite its speed, Geodistpy maintains accuracy on par with Geographiclib, ensuring that fast calculations do not compromise precision.

By adopting Geodistpy, you can significantly enhance the efficiency and performance of your geospatial projects. It is a valuable tool for geospatial professionals and developers seeking both speed and accuracy in their distance computations.

To get started with Geodistpy, visit the [Geodistpy](https://github.com/pawangeek/geodistpy) and explore the documentation for comprehensive usage instructions.
