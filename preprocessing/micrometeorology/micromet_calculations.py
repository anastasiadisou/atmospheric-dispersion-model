import csv
import math
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
YEAR = 2024

# Repository root:
# atmospheric-dispersion-model/
# ├── input/land_use/
# ├── processed/
# └── preprocessing/micrometeorology/micromet_calculations.py
BASE_DIR = Path(__file__).resolve().parents[2]

LANDUSE_DIR = BASE_DIR / "input" / "land_use"
PROCESSED_DIR = BASE_DIR / "processed"

# Inputs produced by the meteorological preprocessing.
INPUT = PROCESSED_DIR / "surface_from_isd.csv"
ZI_FILE = PROCESSED_DIR / "zi.csv"

# Land-use input supplied by the user.
LANDUSE_FILE = LANDUSE_DIR / "landuse.asc"

# Outputs used by the dispersion model.
OUTPUT_CSV = PROCESSED_DIR / "micromet_fluxes.csv"
OUTPUT_GRID = PROCESSED_DIR / "micromet_grid.npz"

START_DT = datetime(YEAR, 1, 1, 0, 0, 0)
END_DT = datetime(YEAR + 1, 1, 1, 0, 0, 0)

# Automatically gives 8760 h for a normal year and 8784 h for a leap year.
HOURS_YEAR = int((END_DT - START_DT).total_seconds() // 3600)


# ============================================================
# CONSTANTS
# ============================================================
SIGMA = 5.670374419e-8
CP = 996
G = 9.81
KAPPA = 0.4
R_D = 287.05

Z_REF = 10.0

USTAR_MIN = 0.05
USTAR_MAX = 2.00

WSTAR_MIN = 0.0
WSTAR_MAX = 3.50

Z0_MIN = 0.0002
Z0_MAX = 2.0

# ============================================================
# LAND-USE PARAMETER TABLE
# ============================================================
LU_PARAMS = {
    # GLCC 1 -> CTGPROC 16 -> CALMET output 10 (urban/built-up)
    1: {
        "z0": 1.00,
        "albedo": 0.18,
        "emiss": 0.97,  # not specified by CALMET; retained for radiation
        "bowen": 1.50,
        "soil_heat_flux": 0.25,
        "lai": 0.20,
        "is_water": False,
    },
    # GLCC 2 -> CTGPROC 21 -> CALMET output 20 (cropland/pasture)
    2: {
        "z0": 0.25,
        "albedo": 0.15,
        "emiss": 0.975,
        "bowen": 1.00,
        "soil_heat_flux": 0.15,
        "lai": 3.00,
        "is_water": False,
    },
    # GLCC 6 -> CTGPROC 21 -> CALMET output 20 (cropland/woodland mosaic)
    6: {
        "z0": 0.25,
        "albedo": 0.15,
        "emiss": 0.975,
        "bowen": 1.00,
        "soil_heat_flux": 0.15,
        "lai": 3.00,
        "is_water": False,
    },
    # GLCC 8 -> CTGPROC 32 -> CALMET output 30 (shrubland)
    8: {
        "z0": 0.05,
        "albedo": 0.25,
        "emiss": 0.982,
        "bowen": 1.00,
        "soil_heat_flux": 0.15,
        "lai": 0.50,
        "is_water": False,
    },
    # GLCC 9 -> CTGPROC 33 -> CALMET output 30 (mixed shrub/grass)
    9: {
        "z0": 0.05,
        "albedo": 0.25,
        "emiss": 0.982,
        "bowen": 1.00,
        "soil_heat_flux": 0.15,
        "lai": 0.50,
        "is_water": False,
    },
    # GLCC 16 -> CTGPROC 52 -> CALMET output 51 (water)
    16: {
        "z0": 0.001,
        "albedo": 0.10,
        "emiss": 0.995,
        "bowen": 0.00,
        "soil_heat_flux": 1.00,  # CALMET category value; special handling below
        "lai": 0.00,
        "is_water": True,
    },
}


# ============================================================
# HELPERS
# ============================================================
def safe_float(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def forward_backward_fill(arr):
    arr = arr.copy()

    for i in range(1, len(arr)):
        if np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            arr[i] = arr[i - 1]

    for i in range(len(arr) - 2, -1, -1):
        if np.isnan(arr[i]) and not np.isnan(arr[i + 1]):
            arr[i] = arr[i + 1]

    return arr


def read_ascii_grid(path):
    header = {}
    data_lines = []

    header_keys = {
        "ncols",
        "nrows",
        "xllcorner",
        "yllcorner",
        "xllcenter",
        "yllcenter",
        "cellsize",
        "nodata_value",
    }

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) >= 2 and parts[0].lower() in header_keys:
                header[parts[0].lower()] = float(parts[1])
            else:
                data_lines.append(line)

    data = np.loadtxt(data_lines)

    nodata = header.get("nodata_value", None)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # ASCII rows: north -> south
    # model arrays: [x, y], y south -> north
    grid = np.flipud(data).T

    return grid, header


def air_density(p_hpa, T):
    p_pa = p_hpa * 100.0
    return p_pa / (R_D * T)


def saturation_vapor_pressure_hpa(Tk):
    Tc = Tk - 273.15
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))


def actual_vapor_pressure_hpa_from_Td(Td_K):
    return saturation_vapor_pressure_hpa(Td_K)


# ============================================================
# SOLAR / RADIATION
# ============================================================
def day_of_year_from_hour_index(hour_index):
    return hour_index // 24 + 1


def utc_hour_from_hour_index(hour_index):
    return hour_index % 24


def solar_declination_rad(n):
    return math.radians(23.45) * math.sin(math.radians(360.0 * (284 + n) / 365.0))


def equation_of_time_minutes(n):
    B = math.radians(360.0 * (n - 81) / 364.0)
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def solar_zenith_deg(hour_index, lat_deg, lon_deg):
    n = day_of_year_from_hour_index(hour_index)
    utc_hour = utc_hour_from_hour_index(hour_index)

    dec = solar_declination_rad(n)
    lat = math.radians(lat_deg)

    eot = equation_of_time_minutes(n)

    # UTC + longitude correction
    lst_hours = utc_hour + lon_deg / 15.0 + eot / 60.0
    hra = math.radians(15.0 * (lst_hours - 12.0))

    cosz = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hra)
    cosz = clip(cosz, -1.0, 1.0)

    return math.degrees(math.acos(cosz))


def clear_sky_shortwave(hour_index, lat_deg, lon_deg):
    zen = solar_zenith_deg(hour_index, lat_deg, lon_deg)

    if zen >= 90.0:
        return 0.0

    n = day_of_year_from_hour_index(hour_index)

    s0 = 1361.0
    ecc = 1.0 + 0.033 * math.cos(2.0 * math.pi * n / 365.0)

    mu0 = math.cos(math.radians(zen))
    tau = 0.70

    return s0 * ecc * mu0 * tau


def cloud_corrected_shortwave(K_clear, cloud_fraction):
    if np.isnan(cloud_fraction):
        cloud_fraction = 0.5

    cf = clip(cloud_fraction, 0.0, 1.0)

    factor = 1.0 - 0.75 * (cf**3.4)
    factor = clip(factor, 0.15, 1.0)

    return K_clear * factor


def atmospheric_emissivity(Ta_K, e_hpa, cloud_fraction):
    e_kpa = e_hpa / 10.0

    eps_clear = 1.24 * (e_kpa / Ta_K) ** (1.0 / 7.0)

    if np.isnan(cloud_fraction):
        cloud_fraction = 0.5

    cf = clip(cloud_fraction, 0.0, 1.0)

    eps_sky = eps_clear * (1.0 + 0.22 * cf * cf)

    return clip(eps_sky, 0.5, 1.0)


def compute_net_radiation_grid(
    hour_index,
    lat_deg,
    lon_deg,
    Ta_K,
    Td_K,
    cloud_fraction,
    albedo_grid,
    emiss_grid,
):
    K_clear = clear_sky_shortwave(hour_index, lat_deg, lon_deg)
    K_down = cloud_corrected_shortwave(K_clear, cloud_fraction)

    K_up = albedo_grid * K_down

    e_hpa = actual_vapor_pressure_hpa_from_Td(Td_K)
    eps_sky = atmospheric_emissivity(Ta_K, e_hpa, cloud_fraction)

    L_down = eps_sky * SIGMA * Ta_K**4

    if K_down > 20.0:
        cf = 0.5 if np.isnan(cloud_fraction) else cloud_fraction
        Ts_K = Ta_K + 3.0 * (1.0 - cf)
    else:
        Ts_K = Ta_K - 1.0

    L_up = emiss_grid * SIGMA * Ts_K**4

    Rn = (K_down - K_up) + (L_down - L_up)

    return K_down, K_up, L_down, L_up, Rn, Ts_K


# ============================================================
# MONIN-OBUKHOV
# ============================================================
def psi_m(z_over_L):
    if z_over_L < 0.0:
        x = (1.0 - 16.0 * z_over_L) ** 0.25

        return (
            2.0 * math.log((1.0 + x) / 2.0)
            + math.log((1.0 + x * x) / 2.0)
            - 2.0 * math.atan(x)
            + math.pi / 2.0
        )
    else:
        return -5.0 * z_over_L


def compute_ustar_and_L(U10, Ta_K, rho, H0, z0):
    U10 = max(0.5, float(U10))
    z0 = clip(float(z0), Z0_MIN, min(Z0_MAX, Z_REF * 0.2))

    # Initial neutral estimate
    L = 1.0e6
    ustar = USTAR_MIN

    for _ in range(8):

        # Stability corrections at reference height and roughness height
        psi_z = psi_m(Z_REF / L)
        psi_z0 = psi_m(z0 / L)

        # Monin-Obukhov wind-profile denominator:
        #
        # ln(z/z0) - psi_m(z/L) + psi_m(z0/L)
        denom = math.log(Z_REF / z0) - psi_z + psi_z0

        denom = max(0.5, denom)

        # Friction velocity
        ustar = KAPPA * U10 / denom
        ustar = clip(ustar, USTAR_MIN, USTAR_MAX)

        # Nearly neutral conditions
        if abs(H0) < 1.0e-6:
            return ustar, 1.0e6

        # Update Monin-Obukhov length
        L_new = -(rho * CP * Ta_K * ustar**3) / (KAPPA * G * H0)

        if not math.isfinite(L_new):
            return ustar, 1.0e6

        # Convergence
        if abs(L_new - L) < 1.0:
            return ustar, L_new

        L = L_new

    return ustar, L


def compute_wstar(H0, rho, Ta_K, zi):
    if np.isnan(zi) or zi <= 0:
        return 0.0

    if H0 <= 0.0:
        return 0.0

    wstar = ((G / Ta_K) * (H0 / (rho * CP)) * zi) ** (1.0 / 3.0)

    return clip(wstar, WSTAR_MIN, WSTAR_MAX)


def main():
    required_files = [INPUT, ZI_FILE, LANDUSE_FILE]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required input file(s) not found:\n"
            f"{missing_text}\n"
            "Run the meteorological preprocessing first and place "
            "landuse.asc in input/land_use/."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running micrometeorological calculations for {YEAR}")
    print(f"Surface input : {INPUT}")
    print(f"Mixing height : {ZI_FILE}")
    print(f"Land use      : {LANDUSE_FILE}")
    print(f"Expected hours: {HOURS_YEAR}")

    # ============================================================
    # LOAD LAND USE
    # ============================================================
    landuse_grid, landuse_header = read_ascii_grid(LANDUSE_FILE)
    nx, ny = landuse_grid.shape

    print("Land-use grid shape:", landuse_grid.shape)
    print("Land-use classes:", np.unique(landuse_grid))

    # ============================================================
    # BUILD SURFACE PARAMETER GRIDS
    # ============================================================
    z0_grid = np.full((nx, ny), 0.10, dtype=float)
    albedo_grid = np.full((nx, ny), 0.20, dtype=float)
    emiss_grid = np.full((nx, ny), 0.97, dtype=float)
    bowen_grid = np.full((nx, ny), 1.00, dtype=float)
    soil_heat_flux_grid = np.full((nx, ny), 0.15, dtype=float)
    lai_grid = np.full((nx, ny), 0.50, dtype=float)
    water_mask = np.zeros((nx, ny), dtype=bool)

    unknown_classes = sorted(
        int(v)
        for v in np.unique(landuse_grid[~np.isnan(landuse_grid)])
        if int(v) not in LU_PARAMS
    )
    if unknown_classes:
        raise ValueError(
            f"Land-use classes missing from LU_PARAMS: {unknown_classes}. "
            "Add an explicit CALMET-compatible mapping before running."
        )

    for lu, p in LU_PARAMS.items():
        mask_lu = landuse_grid == lu

        z0_grid[mask_lu] = p["z0"]
        albedo_grid[mask_lu] = p["albedo"]
        emiss_grid[mask_lu] = p["emiss"]
        bowen_grid[mask_lu] = p["bowen"]
        soil_heat_flux_grid[mask_lu] = p["soil_heat_flux"]
        lai_grid[mask_lu] = p["lai"]
        water_mask[mask_lu] = p["is_water"]

    print("z0 min/mean/max:", z0_grid.min(), z0_grid.mean(), z0_grid.max())
    print("albedo min/mean/max:", albedo_grid.min(), albedo_grid.mean(), albedo_grid.max())
    print("Bowen min/mean/max:", bowen_grid.min(), bowen_grid.mean(), bowen_grid.max())
    print(
        "Soil heat flux parameter min/mean/max:",
        soil_heat_flux_grid.min(),
        soil_heat_flux_grid.mean(),
        soil_heat_flux_grid.max(),
    )

    # ============================================================
    # LOAD zi.csv
    # ============================================================
    zi_data = np.genfromtxt(ZI_FILE, delimiter=",", names=True)
    zi_full = zi_data["zi_m"].astype(float)

    if len(zi_full) != HOURS_YEAR:
        raise ValueError(f"{ZI_FILE} has {len(zi_full)} rows, expected {HOURS_YEAR}.")

    # ============================================================
    # LOAD surface_from_isd.csv
    # ============================================================
    u10_full = np.full(HOURS_YEAR, np.nan)
    Ta_full = np.full(HOURS_YEAR, np.nan)
    Td_full = np.full(HOURS_YEAR, np.nan)
    p_full = np.full(HOURS_YEAR, np.nan)
    cf_full = np.full(HOURS_YEAR, np.nan)
    lat_full = np.full(HOURS_YEAR, np.nan)
    lon_full = np.full(HOURS_YEAR, np.nan)

    with open(INPUT, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)

        print("surface_from_isd columns:")
        print(reader.fieldnames)

        for row in reader:
            h = row.get("hour_index")

            if h is None or h == "":
                continue

            h = int(float(h))

            if h < 0 or h >= HOURS_YEAR:
                continue

            u10_full[h] = safe_float(row.get("u10_mps"))
            Ta_full[h] = safe_float(row.get("Ta_K"))
            Td_full[h] = safe_float(row.get("Td_K"))
            p_full[h] = safe_float(row.get("slp_hPa"))
            cf_full[h] = safe_float(row.get("cloud_fraction"))
            lat_full[h] = safe_float(row.get("lat_deg"))
            lon_full[h] = safe_float(row.get("lon_deg"))

    # ============================================================
    # FILL MISSING
    # ============================================================
    u10_full = forward_backward_fill(u10_full)
    Ta_full = forward_backward_fill(Ta_full)
    Td_full = forward_backward_fill(Td_full)
    p_full = forward_backward_fill(p_full)
    cf_full = forward_backward_fill(cf_full)
    lat_full = forward_backward_fill(lat_full)
    lon_full = forward_backward_fill(lon_full)

    required_arrays = {
        "u10_mps": u10_full,
        "Ta_K": Ta_full,
        "Td_K": Td_full,
        "slp_hPa": p_full,
        "lat_deg": lat_full,
        "lon_deg": lon_full,
    }

    for name, arr in required_arrays.items():
        if np.isnan(arr).any():
            raise ValueError(f"{name} still contains NaNs after fill.")

    # ============================================================
    # ALLOCATE GRID ARRAYS
    # ============================================================
    ustar_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)
    L_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)
    wstar_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)

    H0_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)
    LE_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)
    G0_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)
    Rn_grid = np.zeros((HOURS_YEAR, nx, ny), dtype=np.float32)

    ustar_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    L_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    wstar_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)

    rho_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Kdown_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Kup_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Ldown_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Lup_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Rn_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    Ts_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    H0_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    LE_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)
    G_mean_full = np.zeros(HOURS_YEAR, dtype=np.float32)

    # ============================================================
    # MAIN CALCULATION
    # ============================================================
    for h in range(HOURS_YEAR):
        U10 = float(u10_full[h])
        Ta_K = float(Ta_full[h])
        Td_K = float(Td_full[h])
        p_hPa = float(p_full[h])
        cf = float(cf_full[h]) if not np.isnan(cf_full[h]) else np.nan
        lat_deg = float(lat_full[h])
        lon_deg = float(lon_full[h])
        zi = float(zi_full[h])

        rho = air_density(p_hPa, Ta_K)

        Kd, Ku_grid, Ld, Lu_grid, Rn, Ts_K = compute_net_radiation_grid(
            h,
            lat_deg,
            lon_deg,
            Ta_K,
            Td_K,
            cf,
            albedo_grid,
            emiss_grid,
        )

        # CALMET-compatible surface-energy partitioning.
        #
        # Land:
        #   G  = c_g Rn
        #   H/LE = Bowen
        #   H + LE = Rn - G
        #
        # Water:
        #   The MAKEGEO value soil_heat_flux=1.0 is a CALMET category
        #   parameter and is not interpreted literally as G=Rn here.
        #   Instead, water is handled separately with G=0 and Bowen=0,
        #   so the residual net radiation is assigned to latent heat.
        G0 = soil_heat_flux_grid * Rn
        G0 = np.where(water_mask, 0.0, G0)

        avail = Rn - G0
        LE = np.empty_like(avail, dtype=float)
        H0 = np.empty_like(avail, dtype=float)

        land_mask = ~water_mask
        LE[land_mask] = avail[land_mask] / (1.0 + bowen_grid[land_mask])
        H0[land_mask] = bowen_grid[land_mask] * LE[land_mask]

        LE[water_mask] = avail[water_mask]
        H0[water_mask] = 0.0

        wstar_val_grid = np.zeros((nx, ny), dtype=float)
        ustar_val_grid = np.zeros((nx, ny), dtype=float)
        L_val_grid = np.zeros((nx, ny), dtype=float)

        for i in range(nx):
            for j in range(ny):
                ustar, Lmo = compute_ustar_and_L(
                    U10=U10,
                    Ta_K=Ta_K,
                    rho=rho,
                    H0=float(H0[i, j]),
                    z0=float(z0_grid[i, j]),
                )

                wstar = compute_wstar(
                    H0=float(H0[i, j]),
                    rho=rho,
                    Ta_K=Ta_K,
                    zi=zi,
                )

                ustar_val_grid[i, j] = ustar
                L_val_grid[i, j] = Lmo
                wstar_val_grid[i, j] = wstar

        ustar_grid[h] = ustar_val_grid
        L_grid[h] = L_val_grid
        wstar_grid[h] = wstar_val_grid

        H0_grid[h] = H0
        LE_grid[h] = LE
        G0_grid[h] = G0
        Rn_grid[h] = Rn

        ustar_mean_full[h] = float(np.mean(ustar_val_grid))
        L_mean_full[h] = float(np.mean(L_val_grid))
        wstar_mean_full[h] = float(np.mean(wstar_val_grid))

        rho_full[h] = rho
        Kdown_full[h] = Kd
        Kup_mean_full[h] = float(np.mean(Ku_grid))
        Ldown_full[h] = Ld
        Lup_mean_full[h] = float(np.mean(Lu_grid))
        Rn_mean_full[h] = float(np.mean(Rn))
        Ts_full[h] = Ts_K
        H0_mean_full[h] = float(np.mean(H0))
        LE_mean_full[h] = float(np.mean(LE))
        G_mean_full[h] = float(np.mean(G0))

        if h == 0 or (h + 1) % 1000 == 0 or h == HOURS_YEAR - 1:
            print(
                f"Hour {h + 1}/{HOURS_YEAR} | "
                f"U10={U10:.2f} m/s | "
                f"Rn_mean={np.mean(Rn):.1f} W/m2 | "
                f"ustar_mean={ustar_val_grid.mean():.3f} m/s | "
                f"wstar_mean={wstar_val_grid.mean():.3f} m/s"
            )

    # ============================================================
    # SAVE GRID OUTPUT
    # ============================================================
    np.savez_compressed(
        OUTPUT_GRID,
        landuse_grid=landuse_grid.astype(np.float32),
        z0_grid=z0_grid.astype(np.float32),
        albedo_grid=albedo_grid.astype(np.float32),
        emiss_grid=emiss_grid.astype(np.float32),
        bowen_grid=bowen_grid.astype(np.float32),
        soil_heat_flux_grid=soil_heat_flux_grid.astype(np.float32),
        lai_grid=lai_grid.astype(np.float32),
        water_mask=water_mask,
        ustar_grid=ustar_grid,
        L_grid=L_grid,
        wstar_grid=wstar_grid,
        H0_grid=H0_grid,
        LE_grid=LE_grid,
        G0_grid=G0_grid,
        Rn_grid=Rn_grid,
    )

    print(f"OK: wrote {OUTPUT_GRID}")

    # ============================================================
    # SAVE CSV OUTPUT: DOMAIN-MEAN MICROMET
    # ============================================================
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        fieldnames = [
            "hour_index",
            "u10_mps",
            "Ta_K",
            "Td_K",
            "slp_hPa",
            "cloud_fraction",
            "lat_deg",
            "lon_deg",
            "zi_m",
            "rho",
            "Kdown_Wm2",
            "Kup_Wm2",
            "Ldown_Wm2",
            "Lup_Wm2",
            "Rn_Wm2",
            "Ts_K",
            "H0_Wm2",
            "LE_Wm2",
            "G_Wm2",
            "ustar_mps",
            "L_m",
            "wstar_mps",
        ]

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for h in range(HOURS_YEAR):
            cf = float(cf_full[h]) if not np.isnan(cf_full[h]) else np.nan

            writer.writerow(
                {
                    "hour_index": h,
                    "u10_mps": float(u10_full[h]),
                    "Ta_K": float(Ta_full[h]),
                    "Td_K": float(Td_full[h]),
                    "slp_hPa": float(p_full[h]),
                    "cloud_fraction": cf,
                    "lat_deg": float(lat_full[h]),
                    "lon_deg": float(lon_full[h]),
                    "zi_m": float(zi_full[h]),
                    "rho": float(rho_full[h]),
                    "Kdown_Wm2": float(Kdown_full[h]),
                    "Kup_Wm2": float(Kup_mean_full[h]),
                    "Ldown_Wm2": float(Ldown_full[h]),
                    "Lup_Wm2": float(Lup_mean_full[h]),
                    "Rn_Wm2": float(Rn_mean_full[h]),
                    "Ts_K": float(Ts_full[h]),
                    "H0_Wm2": float(H0_mean_full[h]),
                    "LE_Wm2": float(LE_mean_full[h]),
                    "G_Wm2": float(G_mean_full[h]),
                    "ustar_mps": float(ustar_mean_full[h]),
                    "L_m": float(L_mean_full[h]),
                    "wstar_mps": float(wstar_mean_full[h]),
                }
            )

    print(f"OK: wrote {OUTPUT_CSV}")
    print("Rows written:", HOURS_YEAR)


if __name__ == "__main__":
    main()
