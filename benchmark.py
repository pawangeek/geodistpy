"""
Thorough benchmark comparison: geodistpy vs geopy vs geographiclib

Tests:
1. Single pair distance computation (point-to-point)
2. Pairwise distance matrix (N×N) for various N
3. Accuracy comparison against geographiclib (reference)
4. Edge cases: antipodal points, same point, poles, equator, short/long distances
5. Scaling benchmark
6. Great Circle vs Geodesic trade-off
"""

import time
import numpy as np

# ── Libraries under test ─────────────────────────────────────────────
from geopy.distance import geodesic as geodesic_geopy
from geographiclib.geodesic import Geodesic as geodesic_gglib
from geodistpy.geodesic import (
    geodesic_vincenty,
    geodesic_vincenty_inverse,
    great_circle,
)
from geodistpy.distance import geodist, geodist_matrix

# ── Helpers ──────────────────────────────────────────────────────────


def random_coords(n, seed=42):
    rng = np.random.default_rng(seed)
    lats = rng.uniform(-90, 90, n)
    lons = rng.uniform(-180, 180, n)
    return list(zip(lats, lons))


def time_func(func, *args, repeats=3, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), result


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


# ── Warm-up JIT ─────────────────────────────────────────────────────

print("=" * 80)
print("GEODISTPY BENCHMARK — THOROUGH COMPARISON")
print("=" * 80)
print()
print("Warming up Numba JIT compilation...")
_ = geodesic_vincenty((0, 0), (1, 1))
_ = great_circle((0, 0), (1, 1))
print("JIT warm-up complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: SINGLE PAIR DISTANCE (POINT-TO-POINT)
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 1: Single Pair Distance Computation (best of 3, over 10,000 calls)")
print("─" * 80)

coord1 = (52.5200, 13.4050)  # Berlin
coord2 = (48.8566, 2.3522)  # Paris
N_single = 10_000


def run_geopy_single(n):
    for _ in range(n):
        geodesic_geopy(coord1, coord2).meters


def run_gglib_single(n):
    for _ in range(n):
        geodesic_gglib.WGS84.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])["s12"]


def run_geodistpy_single(n):
    for _ in range(n):
        geodesic_vincenty(coord1, coord2)


t_geopy, _ = time_func(run_geopy_single, N_single, repeats=3)
t_gglib, _ = time_func(run_gglib_single, N_single, repeats=3)
t_gdpy, _ = time_func(run_geodistpy_single, N_single, repeats=3)

print(f"  {'Library':<30} {'Total Time':>12}  {'Per Call':>12}")
print(f"  {'─'*30} {'─'*12}  {'─'*12}")
print(f"  {'Geopy':<30} {fmt_time(t_geopy):>12}  {fmt_time(t_geopy/N_single):>12}")
print(
    f"  {'Geographiclib':<30} {fmt_time(t_gglib):>12}  {fmt_time(t_gglib/N_single):>12}"
)
print(
    f"  {'Geodistpy (Vincenty+Numba)':<30} {fmt_time(t_gdpy):>12}  {fmt_time(t_gdpy/N_single):>12}"
)
print()
print(f"  Geodistpy is {t_geopy/t_gdpy:.1f}x faster than Geopy")
print(f"  Geodistpy is {t_gglib/t_gdpy:.1f}x faster than Geographiclib")
print()


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: PAIRWISE DISTANCE MATRIX (N×N)
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 2: Pairwise Distance Matrix (N×N) — All pairs among N random points")
print("─" * 80)

for N in [50, 100, 200]:
    coords = random_coords(N)
    n_pairs = N * (N - 1) // 2
    print(f"\n  N = {N} points ({n_pairs:,} unique pairs)")
    print(f"  {'Library':<30} {'Time':>12}  {'vs Geopy':>10}")
    print(f"  {'─'*30} {'─'*12}  {'─'*10}")

    def run_geopy_matrix():
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(geodesic_geopy(coords[i], coords[j]).meters)
        return dists

    def run_gglib_matrix():
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(
                    geodesic_gglib.WGS84.Inverse(
                        coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                    )["s12"]
                )
        return dists

    def run_geodistpy_matrix():
        return geodist_matrix(coords, metric="meter")

    t_geopy_m, _ = time_func(run_geopy_matrix, repeats=1)
    t_gglib_m, _ = time_func(run_gglib_matrix, repeats=1)
    t_gdpy_m, _ = time_func(run_geodistpy_matrix, repeats=1)

    print(f"  {'Geopy':<30} {fmt_time(t_geopy_m):>12}  {'(baseline)':>10}")
    print(
        f"  {'Geographiclib':<30} {fmt_time(t_gglib_m):>12}  {t_geopy_m/t_gglib_m:.1f}x"
    )
    print(
        f"  {'Geodistpy (matrix)':<30} {fmt_time(t_gdpy_m):>12}  {t_geopy_m/t_gdpy_m:.1f}x"
    )
    print(
        f"  → Geodistpy is {t_geopy_m/t_gdpy_m:.0f}x faster than Geopy, {t_gglib_m/t_gdpy_m:.0f}x faster than Geographiclib"
    )

print()


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: ACCURACY COMPARISON
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 3: Accuracy Comparison (Geographiclib as reference, 5000 random pairs)")
print("─" * 80)

N_acc = 5000
coords_a = random_coords(N_acc, seed=100)
coords_b = random_coords(N_acc, seed=200)

dists_gglib_arr = []
dists_gdpy_arr = []
dists_geopy_arr = []
dists_gc_arr = []

for i in range(N_acc):
    p1, p2 = coords_a[i], coords_b[i]
    d_ref = geodesic_gglib.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])["s12"]
    d_gdpy = geodesic_vincenty(p1, p2)
    d_geopy = geodesic_geopy(p1, p2).meters
    d_gc = great_circle(p1, p2)
    dists_gglib_arr.append(d_ref)
    dists_gdpy_arr.append(d_gdpy)
    dists_geopy_arr.append(d_geopy)
    dists_gc_arr.append(d_gc)

dists_gglib_arr = np.array(dists_gglib_arr)
dists_gdpy_arr = np.array(dists_gdpy_arr)
dists_geopy_arr = np.array(dists_geopy_arr)
dists_gc_arr = np.array(dists_gc_arr)

err_gdpy = np.abs(dists_gdpy_arr - dists_gglib_arr)
err_geopy = np.abs(dists_geopy_arr - dists_gglib_arr)
err_gc = np.abs(dists_gc_arr - dists_gglib_arr)

rel_err_gdpy = err_gdpy / np.maximum(dists_gglib_arr, 1e-10)
rel_err_geopy = err_geopy / np.maximum(dists_gglib_arr, 1e-10)
rel_err_gc = err_gc / np.maximum(dists_gglib_arr, 1e-10)

print(
    f"\n  {'Method':<30} {'Mean Err (m)':>14} {'Max Err (m)':>14} {'Mean Rel Err':>14} {'Max Rel Err':>14}"
)
print(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*14} {'─'*14}")
print(
    f"  {'Geodistpy (Vincenty)':<30} {err_gdpy.mean():>14.6f} {err_gdpy.max():>14.6f} {rel_err_gdpy.mean():>14.2e} {rel_err_gdpy.max():>14.2e}"
)
print(
    f"  {'Geopy (geodesic)':<30} {err_geopy.mean():>14.6f} {err_geopy.max():>14.6f} {rel_err_geopy.mean():>14.2e} {rel_err_geopy.max():>14.2e}"
)
print(
    f"  {'Geodistpy (Great Circle)':<30} {err_gc.mean():>14.2f} {err_gc.max():>14.2f} {rel_err_gc.mean():>14.2e} {rel_err_gc.max():>14.2e}"
)
print()


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: EDGE CASES & SPECIAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 4: Edge Cases & Special Scenarios")
print("─" * 80)

edge_cases = [
    ("Same point", (52.5200, 13.4050), (52.5200, 13.4050)),
    ("North Pole → South Pole", (90, 0), (-90, 0)),
    ("Antipodal (equator)", (0, 0), (0, 180)),
    ("Near-antipodal", (0.5, 0), (-0.5, 179.9)),
    ("Very short (~1m)", (52.5200, 13.4050), (52.52001, 13.4050)),
    ("Along equator (90°)", (0, 0), (0, 90)),
    ("Along meridian (45°)", (0, 0), (45, 0)),
    ("Cross date line", (0, 179.9), (0, -179.9)),
    ("High latitude (Arctic)", (89.99, 0), (89.99, 180)),
    ("Sydney → New York", (-33.8688, 151.2093), (40.7128, -74.0060)),
    ("London → Tokyo", (51.5074, -0.1278), (35.6762, 139.6503)),
    ("Cape Town → Buenos Aires", (-33.9249, 18.4241), (-34.6037, -58.3816)),
    ("Mumbai → São Paulo", (19.0760, 72.8777), (-23.5505, -46.6333)),
]

print(
    f"\n  {'Scenario':<30} {'Geographiclib':>16} {'Geodistpy':>16} {'Geopy':>16} {'Δ gdpy (m)':>12}"
)
print(f"  {'─'*30} {'─'*16} {'─'*16} {'─'*16} {'─'*12}")

for name, p1, p2 in edge_cases:
    d_ref = geodesic_gglib.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])["s12"]
    d_gdpy = geodesic_vincenty(p1, p2)
    d_geopy = geodesic_geopy(p1, p2).meters
    delta = abs(d_gdpy - d_ref)
    print(
        f"  {name:<30} {d_ref:>16.3f} {d_gdpy:>16.3f} {d_geopy:>16.3f} {delta:>12.6f}"
    )

print()


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: SCALING BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 5: Scaling — Sequential point-to-point calls")
print("─" * 80)

for N in [1_000, 10_000, 50_000]:
    cx = random_coords(N, seed=10)
    cy = random_coords(N, seed=20)

    def run_gdpy():
        for i in range(N):
            geodesic_vincenty(cx[i], cy[i])

    def run_gglib():
        for i in range(N):
            geodesic_gglib.WGS84.Inverse(cx[i][0], cx[i][1], cy[i][0], cy[i][1])["s12"]

    t_g, _ = time_func(run_gdpy, repeats=1)

    if N <= 10_000:
        t_r, _ = time_func(run_gglib, repeats=1)
        print(
            f"  N={N:>6,}: Geodistpy={fmt_time(t_g):>10}  Geographiclib={fmt_time(t_r):>10}  Speedup={t_r/t_g:.1f}x"
        )
    else:
        print(f"  N={N:>6,}: Geodistpy={fmt_time(t_g):>10}  Geographiclib=(skipped)")

print()


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: GREAT CIRCLE vs GEODESIC TRADE-OFF
# ═══════════════════════════════════════════════════════════════════════

print("─" * 80)
print("TEST 6: Great Circle vs Geodesic — Speed/Accuracy Trade-off")
print("─" * 80)

N_gc = 10_000
ca = random_coords(N_gc, seed=300)
cb = random_coords(N_gc, seed=400)


def run_gc():
    for i in range(N_gc):
        great_circle(ca[i], cb[i])


def run_vin():
    for i in range(N_gc):
        geodesic_vincenty(ca[i], cb[i])


t_gc, _ = time_func(run_gc, repeats=3)
t_vin, _ = time_func(run_vin, repeats=3)

print(f"\n  {N_gc:,} sequential calls:")
print(f"  {'Great Circle (Numba)':<30} {fmt_time(t_gc):>12}")
print(f"  {'Vincenty Geodesic (Numba)':<30} {fmt_time(t_vin):>12}")
print(f"  Great Circle is {t_vin/t_gc:.1f}x faster than Vincenty")
print()
print(f"  Accuracy trade-off:")
print(f"  • Great Circle: mean error = {err_gc.mean():.2f}m, max = {err_gc.max():.2f}m")
print(
    f"  • Vincenty:     mean error = {err_gdpy.mean():.6f}m, max = {err_gdpy.max():.6f}m"
)
print()


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
  Single-pair performance (10,000 calls):
  • Geodistpy is {t_geopy/t_gdpy:.0f}x faster than Geopy
  • Geodistpy is {t_gglib/t_gdpy:.0f}x faster than Geographiclib
  • Per-call: Geodistpy ~{t_gdpy/N_single*1e6:.1f}µs vs Geopy ~{t_geopy/N_single*1e6:.0f}µs vs Geographiclib ~{t_gglib/N_single*1e6:.0f}µs

  Accuracy (vs Geographiclib reference, {N_acc} pairs):
  • Geodistpy Vincenty:     mean err = {err_gdpy.mean():.6f}m, max = {err_gdpy.max():.6f}m
  • Geopy:                  mean err = {err_geopy.mean():.6f}m, max = {err_geopy.max():.6f}m
  • Great Circle (approx):  mean err = {err_gc.mean():.1f}m,    max = {err_gc.max():.1f}m

  Great Circle vs Vincenty:
  • Great Circle is {t_vin/t_gc:.1f}x faster but with ~{err_gc.mean():.0f}m average error
  • Use Vincenty for precision, Great Circle for maximum speed with acceptable approximation
""")
