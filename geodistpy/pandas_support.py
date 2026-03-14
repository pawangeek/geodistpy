"""Pandas and GeoPandas support for geodistpy.

Extract (lat, lon) coordinates from pandas DataFrames or GeoPandas GeoDataFrames
so you can use geodistpy functions with tabular data. Requires optional deps:
  pip install geodistpy[pandas]      # DataFrame with lat/lon columns
  pip install geodistpy[geopandas]   # GeoDataFrame with point geometry
"""

from __future__ import annotations

import numpy as np

# Optional dependencies: fail gracefully when not installed
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def coordinates_from_df(df, lat_col=None, lon_col=None):
    """
    Extract (latitude, longitude) as an (n, 2) array from a DataFrame or GeoDataFrame.

    Use this to pass tabular data into geodistpy functions:

    - **pandas DataFrame:** Provide column names with *lat_col* and *lon_col*,
      or leave both None to auto-detect from common names (``lat``/``lon`` or
      ``latitude``/``longitude``).
    - **GeoPandas GeoDataFrame:** Ignores *lat_col*/*lon_col*. Extracts coordinates
      from the ``geometry`` column (assumes point geometry in a geographic CRS,
      e.g. WGS84: x=lon, y=lat).

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Table with coordinate data (either lat/lon columns or point geometry).
    lat_col : str, optional
        Name of the latitude column (DataFrame only). If None, auto-detect.
    lon_col : str, optional
        Name of the longitude column (DataFrame only). If None, auto-detect.

    Returns
    -------
    coords : np.ndarray, shape (n, 2)
        Coordinates as (latitude, longitude) in degrees, one row per input row.
    index : pandas.Index or None
        The DataFrame index, for aligning results (e.g. as Series index).
        None if the input was not a DataFrame/GeoDataFrame.

    Raises
    ------
    ImportError
        If *df* is a DataFrame but pandas is not installed, or a GeoDataFrame
        but geopandas is not installed. Install with
        ``pip install geodistpy[pandas]`` or ``pip install geodistpy[geopandas]``.
    ValueError
        If column names cannot be resolved or geometry is not point type.

    Examples
    --------
    >>> import pandas as pd
    >>> from geodistpy import coordinates_from_df, geodist_to_many
    >>> df = pd.DataFrame({"lat": [48.85, 51.50], "lon": [2.35, -0.12], "name": ["Paris", "London"]})
    >>> coords, index = coordinates_from_df(df)
    >>> coords
    array([[48.85,  2.35],
           [51.5 , -0.12]])
    >>> dists = geodist_to_many((52.52, 13.40), coords, metric="km")
    >>> pd.Series(dists, index=index)
    0    878.39...
    1    932.06...
    dtype: float64
    """
    if gpd is not None and isinstance(df, gpd.GeoDataFrame):
        geom = df.geometry
        if geom is None or len(geom) == 0:
            raise ValueError("GeoDataFrame has no geometry column or is empty")
        # Shapely point: .x = longitude, .y = latitude (in WGS84)
        lats = np.array([g.y for g in geom], dtype=np.float64)
        lons = np.array([g.x for g in geom], dtype=np.float64)
        coords = np.column_stack([lats, lons])
        return coords, df.index

    if pd is not None and isinstance(df, pd.DataFrame):
        if lat_col is not None and lon_col is not None:
            lat_col, lon_col = str(lat_col), str(lon_col)
            if lat_col not in df.columns:
                raise ValueError(f"lat_col '{lat_col}' not in DataFrame columns: {list(df.columns)}")
            if lon_col not in df.columns:
                raise ValueError(f"lon_col '{lon_col}' not in DataFrame columns: {list(df.columns)}")
        else:
            # Auto-detect common column names (lat before latitude, lon before longitude)
            for la, lo in [("lat", "lon"), ("latitude", "longitude"), ("Lat", "Lon"), ("LAT", "LON")]:
                if la in df.columns and lo in df.columns:
                    lat_col, lon_col = la, lo
                    break
            else:
                raise ValueError(
                    "Could not infer lat/lon columns. Provide lat_col and lon_col, "
                    "or use columns named 'lat'/'lon' or 'latitude'/'longitude'."
                )
        coords = np.column_stack([df[lat_col].values.astype(np.float64), df[lon_col].values.astype(np.float64)])
        return coords, df.index

    # Not a DataFrame/GeoDataFrame
    if hasattr(df, "iloc") and hasattr(df, "columns"):
        raise ImportError(
            "pandas is required to use DataFrame input. Install with: pip install geodistpy[pandas]"
        )
    if hasattr(df, "geometry"):
        raise ImportError(
            "geopandas is required to use GeoDataFrame input. Install with: pip install geodistpy[geopandas]"
        )
    raise TypeError(
        "coordinates_from_df expects a pandas DataFrame or GeoPandas GeoDataFrame, "
        f"got {type(df).__name__}"
    )


def _as_coords(points, lat_col=None, lon_col=None):
    """
    Convert a points-like argument to (coords array, index or None).

    - If points is a DataFrame/GeoDataFrame: use coordinates_from_df, return (coords, index).
    - Otherwise: return (np.asarray(points).reshape(-1, 2), None).

    Used internally so that geodist_to_many, geodesic_knn, point_in_radius can
    accept either array-like or DataFrame/GeoDataFrame.
    """
    if (pd is not None and isinstance(points, pd.DataFrame)) or (
        gpd is not None and isinstance(points, gpd.GeoDataFrame)
    ):
        return coordinates_from_df(points, lat_col=lat_col, lon_col=lon_col)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    elif arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2) or be a DataFrame/GeoDataFrame")
    return arr, None
