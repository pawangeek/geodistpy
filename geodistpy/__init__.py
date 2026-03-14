"""geodistpy: Fast geodesic distance calculations on an ellipsoid (WGS-84 default).

Multiple ellipsoid models are supported via the :data:`ELLIPSOIDS` dictionary
or custom ``(a, f)`` tuples.  All public functions default to WGS-84.
"""

from geodistpy.distance import *
from geodistpy.distance import (
    bearing,
    destination,
    interpolate,
    midpoint,
    point_in_radius,
    geodesic_knn,
    geodist_to_many,
)
from geodistpy.geodesic import ELLIPSOIDS
from geodistpy.pandas_support import coordinates_from_df
