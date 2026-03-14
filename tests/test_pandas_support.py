"""Thorough tests for geodistpy pandas and GeoPandas support."""

from __future__ import annotations

import numpy as np
import pytest

from geodistpy import (
    coordinates_from_df,
    geodist,
    geodist_to_many,
    geodesic_knn,
    point_in_radius,
)
import geodistpy.pandas_support as pandas_support
from geodistpy.pandas_support import _as_coords

# ---------------------------------------------------------------------------
# coordinates_from_df — pandas DataFrame
# ---------------------------------------------------------------------------


class TestCoordinatesFromDfDataFrame:
    """Tests for coordinates_from_df with pandas DataFrame (requires pandas)."""

    def test_auto_detect_lat_lon(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.85, 51.50], "lon": [2.35, -0.12], "name": ["Paris", "London"]}
        )
        coords, index = coordinates_from_df(df)
        assert coords.shape == (2, 2)
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])
        assert list(index) == [0, 1]

    def test_auto_detect_latitude_longitude(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"latitude": [48.85, 51.50], "longitude": [2.35, -0.12]})
        coords, _ = coordinates_from_df(df)
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])

    def test_auto_detect_prefers_lat_lon_over_latitude_longitude(self):
        pd = pytest.importorskip("pandas")
        # If both sets exist, first in list wins: ("lat", "lon")
        df = pd.DataFrame(
            {
                "lat": [1.0],
                "lon": [2.0],
                "latitude": [10.0],
                "longitude": [20.0],
            }
        )
        coords, _ = coordinates_from_df(df)
        np.testing.assert_allclose(coords, [[1.0, 2.0]])

    def test_auto_detect_Lat_Lon(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"Lat": [48.85], "Lon": [2.35]})
        coords, _ = coordinates_from_df(df)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])

    def test_auto_detect_LAT_LON(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"LAT": [48.85], "LON": [2.35]})
        coords, _ = coordinates_from_df(df)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])

    def test_explicit_lat_col_lon_col(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"y": [48.85, 51.50], "x": [2.35, -0.12]})
        coords, _ = coordinates_from_df(df, lat_col="y", lon_col="x")
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])

    def test_explicit_columns_single_row(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"latitude": [48.85], "longitude": [2.35]})
        coords, index = coordinates_from_df(df, lat_col="latitude", lon_col="longitude")
        assert coords.shape == (1, 2)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])
        assert len(index) == 1

    def test_lat_col_missing_raises(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lon": [2.35], "other": [1]})
        with pytest.raises(ValueError, match="lat_col .* not in DataFrame columns"):
            coordinates_from_df(df, lat_col="lat", lon_col="lon")

    def test_lon_col_missing_raises(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [48.85], "other": [1]})
        with pytest.raises(ValueError, match="lon_col .* not in DataFrame columns"):
            coordinates_from_df(df, lat_col="lat", lon_col="lon")

    def test_no_inferrable_columns_raises(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [48.85], "b": [2.35]})
        with pytest.raises(ValueError, match="Could not infer lat/lon columns"):
            coordinates_from_df(df)

    def test_empty_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(columns=["lat", "lon"])
        coords, index = coordinates_from_df(df)
        assert coords.shape == (0, 2)
        assert len(index) == 0

    def test_custom_index_preserved(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.85, 51.50], "lon": [2.35, -0.12]},
            index=["Paris", "London"],
        )
        coords, index = coordinates_from_df(df)
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])
        assert list(index) == ["Paris", "London"]

    def test_integer_index(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [48.85], "lon": [2.35]}, index=[100])
        coords, index = coordinates_from_df(df)
        assert list(index) == [100]

    def test_float_columns(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [48.85], "lon": [2.35]})
        coords, _ = coordinates_from_df(df)
        assert coords.dtype == np.float64

    def test_int_columns_converted_to_float(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [48], "lon": [2]})
        coords, _ = coordinates_from_df(df)
        assert coords.dtype == np.float64
        np.testing.assert_allclose(coords, [[48.0, 2.0]])

    def test_extra_columns_ignored(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "lat": [48.85, 51.50],
                "lon": [2.35, -0.12],
                "name": ["Paris", "London"],
                "population": [2.1e6, 8.9e6],
            }
        )
        coords, _ = coordinates_from_df(df)
        assert coords.shape == (2, 2)


# ---------------------------------------------------------------------------
# coordinates_from_df — GeoDataFrame (optional geopandas)
# ---------------------------------------------------------------------------
class TestCoordinatesFromDfGeoDataFrame:
    """Tests for coordinates_from_df with GeoPandas GeoDataFrame."""

    def test_geodataframe_point_geometry(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"name": ["Paris", "London"]},
            geometry=[Point(2.35, 48.85), Point(-0.12, 51.50)],
            crs="EPSG:4326",
        )
        coords, index = coordinates_from_df(gdf)
        assert coords.shape == (2, 2)
        # Point: .x = lon, .y = lat
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])
        assert list(index) == [0, 1]

    def test_geodataframe_custom_index(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"name": ["Paris"]},
            geometry=[Point(2.35, 48.85)],
            index=["city_1"],
        )
        coords, index = coordinates_from_df(gdf)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])
        assert list(index) == ["city_1"]

    def test_geodataframe_lat_col_lon_col_ignored(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"lat": [99.0], "lon": [99.0], "geometry": [Point(2.35, 48.85)]},
            crs="EPSG:4326",
        )
        coords, _ = coordinates_from_df(gdf, lat_col="lat", lon_col="lon")
        # Geometry is used, not lat/lon columns
        np.testing.assert_allclose(coords, [[48.85, 2.35]])

    def test_geodataframe_empty_raises(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[])
        with pytest.raises(ValueError, match="no geometry column or is empty"):
            coordinates_from_df(gdf)


# ---------------------------------------------------------------------------
# coordinates_from_df — invalid input
# ---------------------------------------------------------------------------
class TestCoordinatesFromDfInvalidInput:
    """Invalid inputs to coordinates_from_df."""

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            coordinates_from_df([(48.85, 2.35)])

    def test_ndarray_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            coordinates_from_df(np.array([[48.85, 2.35]]))

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            coordinates_from_df({"lat": [48.85], "lon": [2.35]})

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            coordinates_from_df(None)

    def test_scalar_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            coordinates_from_df(42)


class TestCoordinatesFromDfOptionalDependencyFallbacks:
    """Branches that handle missing optional dependencies."""

    def test_import_error_for_dataframe_like_when_pandas_missing(self, monkeypatch):
        class FakeDataFrame:
            columns = ["lat", "lon"]
            iloc = object()

        monkeypatch.setattr(pandas_support, "pd", None)
        with pytest.raises(ImportError, match="pandas is required"):
            coordinates_from_df(FakeDataFrame())

    def test_import_error_for_geodataframe_like_when_geopandas_missing(
        self, monkeypatch
    ):
        class FakeGeoDataFrame:
            geometry = []

        monkeypatch.setattr(pandas_support, "gpd", None)
        with pytest.raises(ImportError, match="geopandas is required"):
            coordinates_from_df(FakeGeoDataFrame())

    def test_geodataframe_branch_without_geopandas_installed(self, monkeypatch):
        class FakePoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class FakeGeoDataFrame:
            def __init__(self):
                self.geometry = [FakePoint(2.35, 48.85), FakePoint(-0.12, 51.50)]
                self.index = [10, 20]

        class FakeGpd:
            GeoDataFrame = FakeGeoDataFrame

        monkeypatch.setattr(pandas_support, "gpd", FakeGpd)
        coords, index = coordinates_from_df(FakeGeoDataFrame())
        np.testing.assert_allclose(coords, [[48.85, 2.35], [51.50, -0.12]])
        assert list(index) == [10, 20]

    def test_geodataframe_branch_empty_geometry_raises(self, monkeypatch):
        class FakeGeoDataFrame:
            def __init__(self):
                self.geometry = []
                self.index = []

        class FakeGpd:
            GeoDataFrame = FakeGeoDataFrame

        monkeypatch.setattr(pandas_support, "gpd", FakeGpd)
        with pytest.raises(ValueError, match="no geometry column or is empty"):
            coordinates_from_df(FakeGeoDataFrame())


# ---------------------------------------------------------------------------
# _as_coords (internal) — array-like and DataFrame
# ---------------------------------------------------------------------------
class TestAsCoords:
    """Tests for _as_coords used internally by geodist_to_many, geodesic_knn, point_in_radius."""

    def test_array_n2_returns_coords_and_none_index(self):
        arr = np.array([[48.85, 2.35], [51.50, -0.12]])
        coords, index = _as_coords(arr)
        np.testing.assert_array_almost_equal(coords, arr)
        assert index is None

    def test_list_of_tuples(self):
        pts = [(48.85, 2.35), (51.50, -0.12)]
        coords, index = _as_coords(pts)
        assert coords.shape == (2, 2)
        np.testing.assert_allclose(coords, pts)
        assert index is None

    def test_single_point_1d_reshaped_to_1x2(self):
        pt = np.array([48.85, 2.35])
        coords, index = _as_coords(pt)
        assert coords.shape == (1, 2)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])
        assert index is None

    def test_single_point_tuple(self):
        coords, index = _as_coords((48.85, 2.35))
        assert coords.shape == (1, 2)
        assert index is None

    def test_wrong_shape_n3_raises(self):
        arr = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="shape \\(n, 2\\)"):
            _as_coords(arr)

    def test_dataframe_returns_coords_and_index(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [48.85], "lon": [2.35]}, index=[10])
        coords, index = _as_coords(df)
        np.testing.assert_allclose(coords, [[48.85, 2.35]])
        assert index is not None
        assert list(index) == [10]

    def test_dataframe_explicit_lat_lon(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [48.85], "b": [2.35]})
        coords, _ = _as_coords(df, lat_col="a", lon_col="b")
        np.testing.assert_allclose(coords, [[48.85, 2.35]])


# ---------------------------------------------------------------------------
# geodist_to_many with DataFrame
# ---------------------------------------------------------------------------
class TestGeodistToManyDataFrame:
    """geodist_to_many with pandas DataFrame / GeoDataFrame."""

    def test_returns_series_with_dataframe_index(self):
        pd = pytest.importorskip("pandas")
        origin = (52.5200, 13.4050)
        df = pd.DataFrame(
            {"lat": [48.8566, 51.5074], "lon": [2.3522, -0.1278]},
            index=[10, 20],
        )
        result = geodist_to_many(origin, df, metric="km")
        assert hasattr(result, "index")
        assert list(result.index) == [10, 20]
        expected = geodist_to_many(origin, df[["lat", "lon"]].values, metric="km")
        np.testing.assert_allclose(result.values, expected)

    def test_single_row_dataframe(self):
        pd = pytest.importorskip("pandas")
        origin = (52.52, 13.40)
        df = pd.DataFrame({"lat": [48.8566], "lon": [2.3522]})
        result = geodist_to_many(origin, df, metric="km")
        assert len(result) == 1
        np.testing.assert_allclose(
            result.values, geodist(origin, (48.8566, 2.3522), metric="km")
        )

    def test_explicit_lat_col_lon_col(self):
        pd = pytest.importorskip("pandas")
        origin = (52.52, 13.40)
        df = pd.DataFrame({"y": [48.8566], "x": [2.3522]})
        result = geodist_to_many(origin, df, lat_col="y", lon_col="x", metric="km")
        assert len(result) == 1
        np.testing.assert_allclose(
            result.values, geodist(origin, (48.8566, 2.3522), metric="km")
        )

    def test_string_index(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.85], "lon": [2.35]},
            index=["Paris"],
        )
        result = geodist_to_many((52.52, 13.40), df, metric="km")
        assert list(result.index) == ["Paris"]


# ---------------------------------------------------------------------------
# geodesic_knn with DataFrame
# ---------------------------------------------------------------------------
class TestGeodesicKnnDataFrame:
    """geodesic_knn with pandas DataFrame."""

    def test_returns_index_labels_matching_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.8566, 40.7128, 51.5074], "lon": [2.3522, -74.006, -0.1278]},
            index=["Paris", "NYC", "London"],
        )
        idx, dists = geodesic_knn((52.5200, 13.4050), df, k=2, metric="km")
        assert len(idx) == 2
        assert len(dists) == 2
        assert set(idx) == {"Paris", "London"}  # nearest two

    def test_k1_returns_single_label(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.8566, 51.5074], "lon": [2.3522, -0.1278]},
            index=["Paris", "London"],
        )
        idx, dists = geodesic_knn((52.52, 13.40), df, k=1, metric="km")
        assert len(idx) == 1
        assert idx[0] == "Paris"
        assert dists[0] == pytest.approx(
            geodist((52.52, 13.40), (48.8566, 2.3522), metric="km"), rel=1e-5
        )

    def test_k_all_returns_all_sorted(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "lat": [48.8566, 40.7128, 51.5074],
                "lon": [2.3522, -74.006, -0.1278],
            }
        )
        idx, dists = geodesic_knn((52.52, 13.40), df, k=3, metric="km")
        assert len(dists) == 3
        assert list(dists) == sorted(dists)

    def test_explicit_lat_col_lon_col(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"latitude": [48.8566], "longitude": [2.3522]})
        idx, dists = geodesic_knn(
            (52.52, 13.40),
            df,
            k=1,
            metric="km",
            lat_col="latitude",
            lon_col="longitude",
        )
        assert len(idx) == 1
        assert dists[0] == pytest.approx(
            geodist((52.52, 13.40), (48.8566, 2.3522), metric="km"), rel=1e-5
        )


# ---------------------------------------------------------------------------
# point_in_radius with DataFrame
# ---------------------------------------------------------------------------
class TestPointInRadiusDataFrame:
    """point_in_radius with pandas DataFrame."""

    def test_returns_index_labels(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.8566, 40.7128, 51.5074], "lon": [2.3522, -74.006, -0.1278]},
            index=["Paris", "NYC", "London"],
        )
        idx, dists = point_in_radius((52.5200, 13.4050), df, 1000, metric="km")
        assert 2 <= len(idx) <= 3
        assert "Paris" in idx
        assert "London" in idx
        assert "NYC" not in idx

    def test_none_within_returns_empty(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"lat": [40.7128], "lon": [-74.006]})  # NYC
        idx, dists = point_in_radius((52.52, 13.40), df, 10, metric="km")
        assert len(idx) == 0
        assert len(dists) == 0

    def test_all_within_returns_all_labels(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"lat": [48.8566, 51.5074], "lon": [2.3522, -0.1278]},
            index=["Paris", "London"],
        )
        idx, dists = point_in_radius((52.52, 13.40), df, 10000, metric="km")
        assert len(idx) == 2
        assert set(idx) == {"Paris", "London"}

    def test_distances_match_geodist(self):
        pd = pytest.importorskip("pandas")
        center = (52.52, 13.40)
        df = pd.DataFrame({"lat": [48.8566], "lon": [2.3522]})
        idx, dists = point_in_radius(center, df, 1000, metric="km")
        assert len(idx) == 1
        expected = geodist(center, (48.8566, 2.3522), metric="km")
        assert dists[0] == pytest.approx(expected, rel=1e-6)

    def test_explicit_lat_col_lon_col(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"y": [48.8566], "x": [2.3522]})
        idx, dists = point_in_radius(
            (52.52, 13.40), df, 1000, metric="km", lat_col="y", lon_col="x"
        )
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# Round-trip: coordinates_from_df -> geodist_to_many (same as passing DataFrame)
# ---------------------------------------------------------------------------
class TestRoundTrip:
    """Consistency between coordinates_from_df + array API and DataFrame API."""

    def test_geodist_to_many_array_vs_dataframe_same_result(self):
        pd = pytest.importorskip("pandas")
        origin = (52.5200, 13.4050)
        df = pd.DataFrame(
            {
                "lat": [48.8566, 51.5074, 40.7128],
                "lon": [2.3522, -0.1278, -74.006],
            }
        )
        by_df = geodist_to_many(origin, df, metric="km").values
        coords, _ = coordinates_from_df(df)
        by_array = geodist_to_many(origin, coords, metric="km")
        np.testing.assert_allclose(by_df, by_array)

    def test_geodesic_knn_array_vs_dataframe_same_neighbors(self):
        pd = pytest.importorskip("pandas")
        query = (52.52, 13.40)
        df = pd.DataFrame(
            {
                "lat": [48.8566, 40.7128, 51.5074],
                "lon": [2.3522, -74.006, -0.1278],
            }
        )
        idx_df, dist_df = geodesic_knn(query, df, k=2, metric="km")
        coords, index = coordinates_from_df(df)
        idx_arr, dist_arr = geodesic_knn(query, coords, k=2, metric="km")
        np.testing.assert_allclose(dist_df, dist_arr)
        assert list(idx_df) == [index[i] for i in idx_arr]
