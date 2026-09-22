import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
MODEL_YEAR = 2024

# Requested output period. Date-only END_DATE includes the whole day.
START_DATE = "2024-02-01"
END_DATE = "2024-02-01"
SPINUP_DAYS = 1

# Repository paths:
# atmospheric-dispersion-model/
# ├── input/land_use/
# ├── input/emissions/
# ├── processed/
# ├── results/
# └── model/dispersion_model.py
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
LANDUSE_DIR = BASE_DIR / "input" / "land_use"
EMISSIONS_DIR = BASE_DIR / "input" / "emissions"
RESULTS_DIR = BASE_DIR / "results"

WIND_FILE = PROCESSED_DIR / "wind_from_isd.csv"
KZ_FILE = PROCESSED_DIR / "kz.csv"
ZI_FILE = PROCESSED_DIR / "zi.csv"
MICROMET_FILE = PROCESSED_DIR / "micromet_fluxes.csv"
LANDUSE_FILE = LANDUSE_DIR / "landuse.asc"
MICROMET_GRID_FILE = PROCESSED_DIR / "micromet_grid.npz"

# Emissions: choose "constant" or "file".
EMISSION_MODE = "constant"
Q_CONSTANT_G_S = 20.0

# Used only when EMISSION_MODE = "file".
EMISSIONS_FILE = EMISSIONS_DIR / "emissions.csv"
EMISSIONS_COLUMN = "emission_g_s"

# Computational grid
NX = 20
NY = 20
DX_M = 1000.0
DY_M = 1000.0
X_ORIGIN_KM = 710.12584
Y_ORIGIN_KM = 4203.78765
RECEPTOR_HEIGHT_M = 2.0

# Point source
SOURCE_X_M = 719625.84
SOURCE_Y_M = 4213287.65
STACK_HEIGHT_M = 60.0
STACK_DIAMETER_M = 3.0
EXIT_VELOCITY_MPS = 30.0
EXIT_TEMPERATURE_K = 423.15

# Output options
SAVE_TXT = True
SAVE_GRD = True
SHOW_PLOT = True

MODEL_START = datetime(MODEL_YEAR, 1, 1, 0, 0)
MODEL_END = datetime(MODEL_YEAR + 1, 1, 1, 0, 0)
HOURS_YEAR = int((MODEL_END - MODEL_START).total_seconds() // 3600)



def parse_period_date(value, is_end=False):
    """Parse YYYY-MM-DD or YYYY-MM-DD HH:MM into an hourly datetime."""
    value = value.strip()
    formats = ("%Y-%m-%d %H:%M", "%Y-%m-%d")

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d" and is_end:
                # A date-only END_DATE includes the whole ending day.
                dt += timedelta(days=1)
            elif is_end:
                # An explicitly supplied end time is also treated as inclusive.
                dt += timedelta(hours=1)
            return dt
        except ValueError:
            pass

    raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD or YYYY-MM-DD HH:MM.")


REQUESTED_START = parse_period_date(START_DATE, is_end=False)
REQUESTED_END_EXCLUSIVE = parse_period_date(END_DATE, is_end=True)

if REQUESTED_START.minute != 0 or REQUESTED_END_EXCLUSIVE.minute != 0:
    raise ValueError("Simulation limits must fall on complete hours.")

if REQUESTED_START < MODEL_START or REQUESTED_END_EXCLUSIVE > MODEL_END:
    raise ValueError(
        f"Requested period must lie within {MODEL_START:%Y-%m-%d} "
        f"and {(MODEL_END - timedelta(days=1)):%Y-%m-%d}."
    )

if REQUESTED_END_EXCLUSIVE <= REQUESTED_START:
    raise ValueError("END_DATE must be later than or equal to START_DATE.")

OUTPUT_START_HOUR = int((REQUESTED_START - MODEL_START).total_seconds() // 3600)
OUTPUT_END_HOUR = int((REQUESTED_END_EXCLUSIVE - MODEL_START).total_seconds() // 3600)

START_HOUR = max(0, OUTPUT_START_HOUR - SPINUP_DAYS * 24)
END_HOUR = OUTPUT_END_HOUR
HOURS_RUN = END_HOUR - START_HOUR
HOURS_OUTPUT = OUTPUT_END_HOUR - OUTPUT_START_HOUR

# ============================================================
# PERIOD TAG GENERATION
# ============================================================
req_end_incl = REQUESTED_END_EXCLUSIVE - timedelta(hours=1)

if REQUESTED_START.strftime("%Y-%m-%d") == req_end_incl.strftime("%Y-%m-%d"):
    # Ημερήσια προσομοίωση: YYYY-MM-DD
    PERIOD_TAG = f"{REQUESTED_START:%Y-%m-%d}"
elif (
    REQUESTED_START.day == 1
    and REQUESTED_START.hour == 0
    and REQUESTED_END_EXCLUSIVE.day == 1
    and REQUESTED_END_EXCLUSIVE.hour == 0
    and req_end_incl.month == REQUESTED_START.month
    and req_end_incl.year == REQUESTED_START.year
):
    # Μηνιαία προσομοίωση: YYYY-MM
    PERIOD_TAG = f"{REQUESTED_START:%Y-%m}"
else:
    # Περίοδος προσομοίωσης: YYYY-MM-DD_to_YYYY-MM-DD
    PERIOD_TAG = f"{REQUESTED_START:%Y-%m-%d}_to_{req_end_incl:%Y-%m-%d}"

print("Running simulation for selected period:")
print(f"Requested start   : {REQUESTED_START:%Y-%m-%d %H:%M}")
print(
    "Requested end     : "
    f"{REQUESTED_END_EXCLUSIVE - timedelta(hours=1):%Y-%m-%d %H:%M}"
)
print(f"Spin-up days      : {SPINUP_DAYS}")
print(f"First simulated h : {START_HOUR}")
print(f"Last simulated h  : {END_HOUR - 1}")
print(f"Output hours      : {HOURS_OUTPUT}")

# ============================================================
# INPUT / OUTPUT HELPERS
# ============================================================


def read_ascii_grid_int(path):
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

    grid = np.flipud(data).T

    return grid


SEA_CLASS = 16


def sample_grid_at_points(xp, yp, grid, xmin, ymin, dx, dy):
    ix = np.floor((xp - xmin) / dx).astype(int)
    iy = np.floor((yp - ymin) / dy).astype(int)

    inside = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])

    # Use domain-mean fallback for puffs outside the land-use/micromet grid.
    fallback = float(np.nanmean(grid))
    values = np.full(xp.shape, fallback, dtype=float)
    values[inside] = grid[ix[inside], iy[inside]]

    return values, inside


# ============================================================
# VALIDATE INPUTS
# ============================================================
required_files = [
    WIND_FILE,
    KZ_FILE,
    ZI_FILE,
    MICROMET_FILE,
    LANDUSE_FILE,
    MICROMET_GRID_FILE,
]
if EMISSION_MODE.lower() == "file":
    required_files.append(EMISSIONS_FILE)

missing_files = [path for path in required_files if not path.exists()]
if missing_files:
    missing_text = "\n".join(f"  - {path}" for path in missing_files)
    raise FileNotFoundError(f"Required input file(s) not found:\n{missing_text}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load wind from wind_from_isd.csv (u,v,Ta)
# ============================================================
wind = np.genfromtxt(WIND_FILE, delimiter=",", names=True)

hour_index = wind["hour_index"].astype(int)
u_series = wind["u_mps"].astype(float)
v_series = wind["v_mps"].astype(float)
Ta_series = wind["Ta_K"].astype(float)

u_full = np.full(HOURS_YEAR, np.nan, dtype=float)
v_full = np.full(HOURS_YEAR, np.nan, dtype=float)
Ta_full = np.full(HOURS_YEAR, np.nan, dtype=float)

mask = (hour_index >= 0) & (hour_index < HOURS_YEAR)
u_full[hour_index[mask]] = u_series[mask]
v_full[hour_index[mask]] = v_series[mask]
Ta_full[hour_index[mask]] = Ta_series[mask]

# Fill missing hours (forward + backward)
for i in range(HOURS_YEAR):
    if np.isnan(u_full[i]) and i > 0:
        u_full[i] = u_full[i - 1]
        v_full[i] = v_full[i - 1]
    if np.isnan(Ta_full[i]) and i > 0:
        Ta_full[i] = Ta_full[i - 1]

for i in range(HOURS_YEAR - 1, -1, -1):
    if np.isnan(u_full[i]) and i < HOURS_YEAR - 1:
        u_full[i] = u_full[i + 1]
        v_full[i] = v_full[i + 1]
    if np.isnan(Ta_full[i]) and i < HOURS_YEAR - 1:
        Ta_full[i] = Ta_full[i + 1]

# ============================================================
# Kz(t) from kz.csv
# ============================================================
kz_data = np.genfromtxt(KZ_FILE, delimiter=",", names=True)
Kz_full = kz_data["Kz_m2ps"].astype(float)
if len(Kz_full) != HOURS_YEAR:
    raise ValueError(f"kz.csv has {len(Kz_full)} rows, expected {HOURS_YEAR}.")

# ============================================================
# zi(t) from zi.csv
# ============================================================
zi_data = np.genfromtxt(ZI_FILE, delimiter=",", names=True)
zi_full = zi_data["zi_m"].astype(float)
if len(zi_full) != HOURS_YEAR:
    raise ValueError(f"zi.csv has {len(zi_full)} rows, expected {HOURS_YEAR}.")

# ============================================================
# Emissions Q(t)
# ============================================================
if EMISSION_MODE.lower() == "constant":
    Q_g_s_full = np.full(HOURS_YEAR, float(Q_CONSTANT_G_S), dtype=float)

elif EMISSION_MODE.lower() == "file":
    emis_data = np.genfromtxt(
        EMISSIONS_FILE, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )

    available_columns = emis_data.dtype.names or ()
    if "hour_index" not in available_columns:
        raise ValueError(f"{EMISSIONS_FILE} must contain an 'hour_index' column.")
    if EMISSIONS_COLUMN not in available_columns:
        raise ValueError(
            f"{EMISSIONS_FILE} does not contain '{EMISSIONS_COLUMN}'. "
            f"Available columns: {available_columns}"
        )

    emis_hour = emis_data["hour_index"].astype(int)
    emis_q_g_s = emis_data[EMISSIONS_COLUMN].astype(float)

    Q_g_s_full = np.full(HOURS_YEAR, np.nan, dtype=float)
    mask_q = (emis_hour >= 0) & (emis_hour < HOURS_YEAR)
    Q_g_s_full[emis_hour[mask_q]] = emis_q_g_s[mask_q]

    for i in range(1, HOURS_YEAR):
        if np.isnan(Q_g_s_full[i]) and not np.isnan(Q_g_s_full[i - 1]):
            Q_g_s_full[i] = Q_g_s_full[i - 1]

    for i in range(HOURS_YEAR - 2, -1, -1):
        if np.isnan(Q_g_s_full[i]) and not np.isnan(Q_g_s_full[i + 1]):
            Q_g_s_full[i] = Q_g_s_full[i + 1]

    if np.isnan(Q_g_s_full).any():
        raise ValueError(
            f"{EMISSIONS_FILE} still contains missing emission values after fill."
        )
else:
    raise ValueError("EMISSION_MODE must be either 'constant' or 'file'.")


print(
    "Met series ready:",
    HOURS_YEAR,
    "hours",
)

# ============================================================
# Slice only selected day
# ============================================================
u_day = u_full[START_HOUR:END_HOUR]
v_day = v_full[START_HOUR:END_HOUR]
Ta_day = Ta_full[START_HOUR:END_HOUR]
Kz_day = Kz_full[START_HOUR:END_HOUR]
zi_day = zi_full[START_HOUR:END_HOUR]
Q_g_s_day = Q_g_s_full[START_HOUR:END_HOUR]

# ============================================================
# Load scalar micrometeorology from micromet_fluxes.csv
# ============================================================
micromet = np.genfromtxt(MICROMET_FILE, delimiter=",", names=True)

ustar_full = micromet["ustar_mps"].astype(float)
wstar_full = micromet["wstar_mps"].astype(float)
Lmo_full = micromet["L_m"].astype(float)

if len(ustar_full) != HOURS_YEAR:
    raise ValueError("micromet_fluxes.csv length mismatch.")

ustar_day = ustar_full[START_HOUR:END_HOUR]
wstar_day = wstar_full[START_HOUR:END_HOUR]
Lmo_day = Lmo_full[START_HOUR:END_HOUR]

print("Micromet loaded")

if len(u_day) != HOURS_RUN:
    raise ValueError("Selected simulation period has wrong number of hours.")

# ============================================================
# Grid
# ============================================================
nx, ny = NX, NY
dx, dy = DX_M, DY_M

XORIGKM = X_ORIGIN_KM
YORIGKM = Y_ORIGIN_KM
x0_m = XORIGKM * 1000.0
y0_m = YORIGKM * 1000.0

x = x0_m + (np.arange(nx) + 0.5) * dx
y = y0_m + (np.arange(ny) + 0.5) * dy
X, Y = np.meshgrid(x, y, indexing="ij")

z_rec = RECEPTOR_HEIGHT_M
xmin, xmax = x0_m, x0_m + nx * dx
ymin, ymax = y0_m, y0_m + ny * dy

landuse_grid = read_ascii_grid_int(LANDUSE_FILE)


# ============================================================
# SOURCE AND DOMAIN PARAMETERS
# ============================================================
x_src = SOURCE_X_M
y_src = SOURCE_Y_M
H = STACK_HEIGHT_M

ix_src = int(np.floor((x_src - xmin) / dx))
iy_src = int(np.floor((y_src - ymin) / dy))

ix_src = int(np.clip(ix_src, 0, nx - 1))
iy_src = int(np.clip(iy_src, 0, ny - 1))


# ============================================================
# LOAD GRIDDED MICROMETEOROLOGY
# ============================================================
mic = np.load(MICROMET_GRID_FILE)

ustar_grid_full = mic["ustar_grid"]
wstar_grid_full = mic["wstar_grid"]


# Source strength is hourly and is read from EMISSIONS_FILE.
# It is converted to ug/s inside the hourly loop.

# Horizontal diffusivity from stability-based sigma_v and Ly=zi
KXY_MIN = 10.0
KXY_MAX = 1500.0

dt_sub = 300.0
substeps = int(3600 / dt_sub)

SIGMA0_H = 125.0  # initial horizontal spread
SIGMA0_Z = 100.0  # initial vertical spread

max_puffs = 150

buffer_m = 20000.0
sig_cap = 30000.0

ROOT_2PI3 = (2.0 * np.pi) ** 1.5

# Stack
D_stack = STACK_DIAMETER_M
vs = EXIT_VELOCITY_MPS
Ts_K = EXIT_TEMPERATURE_K
p_wind = 0.2

# Puff splitting
SIG_SPLIT = 1500.0
SPLIT_OFFSET_FRAC = 0.30
MAX_PUFFS_AFTER_SPLIT = 2 * max_puffs


# Bounds for turbulence intensities (m/s)
SIGW_MIN = 0.25
SIGW_MAX = 2.00
SIGV_MIN = 0.30
SIGV_MAX = 3.50

# ============================================================
# FUNCTIONS: LAND-USE / GRID HELPERS
# ============================================================


def landuse_to_Kxy_factor(lu_array):
    f = np.ones_like(lu_array, dtype=float)

    # sea → more horizontal spreading
    f[lu_array == 16] = 1.4

    # urban → λίγο πιο "κρατημένο"
    f[lu_array == 1] = 0.8

    # agriculture (6) → baseline
    f[lu_array == 6] = 1.0

    # forest (2)
    f[lu_array == 2] = 0.9

    return f


# ============================================================
# FUNCTIONS: MICROMET / STABILITY / PLUME RISE
# ============================================================


def plume_rise_briggs(Umean, Ta_K, H, D, vs, Ts_K, stability, p=p_wind):
    g = 9.81
    rs = D / 2.0

    # Wind speed at stack height
    Umean = max(0.5, float(Umean))
    uz = max(0.5, Umean * (H / 10.0) ** p)

    # Temperature difference
    dT = max(0.0, float(Ts_K - Ta_K))

    # Buoyancy flux parameter
    F = g * vs * (rs**2) * (dT / Ts_K)

    if F <= 0.0:
        return 0.0

    stability = str(stability).lower()

    # =========================================================
    # STABLE CONDITIONS
    # Briggs stable buoyancy-rise formulation
    # =========================================================
    if stability == "stable":

        # EPA representative potential-temperature gradient
        # for stability class E
        dtheta_dz = 0.020  # K/m

        # Stability parameter [s^-2]
        s = (g / Ta_K) * dtheta_dz

        # Final stable plume rise
        dh = 2.6 * (F / (uz * s)) ** (1.0 / 3.0)

    # =========================================================
    # NEUTRAL / UNSTABLE CONDITIONS
    # Briggs buoyancy-rise formulation
    # =========================================================
    else:

        if F <= 55.0:
            xf = 49.0 * (F ** (5.0 / 8.0))
        else:
            xf = 119.0 * (F ** (2.0 / 5.0))

        dh = 1.6 * (F ** (1.0 / 3.0)) * (xf ** (2.0 / 3.0)) / uz

    return float(max(0.0, dh))


# ============================================================
# FUNCTIONS: CONCENTRATION AND PUFF MANAGEMENT
# ============================================================


def exp_z_with_lid(z_rec, z_p, sigz, zi, n_images=1):
    inv2 = 1.0 / (2.0 * sigz * sigz)
    total = 0.0
    for n in range(-n_images, n_images + 1):
        shift = 2.0 * n * zi
        dz_a = z_rec - (z_p + shift)
        dz_b = z_rec - (-z_p + shift)
        total += np.exp(-(dz_a * dz_a) * inv2) + np.exp(-(dz_b * dz_b) * inv2)
    return total


def compute_C_grid_with_zi(
    X, Y, z_rec, zi, xp, yp, zp, sx, sy, sz, m, block=64, n_images=1
):
    """
    Returns concentration in ug/m3 because:
    - puff mass m is stored in ug
    - sigmas are in m
    """
    C = np.zeros_like(X, dtype=float)
    n = xp.size
    if n == 0:
        return C

    X3 = X[..., None]
    Y3 = Y[..., None]

    for i in range(0, n, block):
        j = slice(i, min(i + block, n))

        dx2 = (X3 - xp[j]) ** 2
        dy2 = (Y3 - yp[j]) ** 2

        inv_sx2 = 1.0 / (sx[j] ** 2)
        inv_sy2 = 1.0 / (sy[j] ** 2)

        exp_xy = np.exp(-0.5 * (dx2 * inv_sx2 + dy2 * inv_sy2))

        exp_z = np.zeros_like(exp_xy, dtype=float)
        for k in range(exp_xy.shape[-1]):
            exp_z[..., k] = exp_z_with_lid(
                z_rec, zp[j][k], sz[j][k], zi, n_images=n_images
            )

        norm = m[j] / (ROOT_2PI3 * sx[j] * sy[j] * sz[j])
        C += (exp_xy * exp_z * norm).sum(axis=-1)

    return C


def split_puffs_crosswind(xp, yp, zp, sx, sy, sz, m, u, v, sig_split, frac):
    """
    Split puffs with sy > sig_split into 2 puffs offset in crosswind direction.
    Conserves mass.
    """
    if xp.size == 0:
        return xp, yp, zp, sx, sy, sz, m

    mask = sy > sig_split
    if not np.any(mask):
        return xp, yp, zp, sx, sy, sz, m

    U = np.sqrt(u * u + v * v)
    if U < 1e-6:
        return xp, yp, zp, sx, sy, sz, m

    ex = u / U
    ey = v / U
    cx = -ey
    cy = ex

    d = frac * sy[mask]
    keep = ~mask

    xp_keep, yp_keep, zp_keep = xp[keep], yp[keep], zp[keep]
    sx_keep, sy_keep, sz_keep = sx[keep], sy[keep], sz[keep]
    m_keep = m[keep]

    xp_s, yp_s, zp_s = xp[mask], yp[mask], zp[mask]
    sx_s, sy_s, sz_s = sx[mask], sy[mask], sz[mask]
    m_s = m[mask] * 0.5

    xp1 = xp_s + cx * d
    yp1 = yp_s + cy * d
    xp2 = xp_s - cx * d
    yp2 = yp_s - cy * d

    xp_new = np.concatenate([xp_keep, xp1, xp2])
    yp_new = np.concatenate([yp_keep, yp1, yp2])
    zp_new = np.concatenate([zp_keep, zp_s, zp_s])

    sx_new = np.concatenate([sx_keep, sx_s, sx_s])
    sy_new = np.concatenate([sy_keep, sy_s, sy_s])
    sz_new = np.concatenate([sz_keep, sz_s, sz_s])

    m_new = np.concatenate([m_keep, m_s, m_s])

    return xp_new, yp_new, zp_new, sx_new, sy_new, sz_new, m_new


# ============================================================
# FUNCTIONS: TURBULENCE / TAYLOR GROWTH / UTILITIES
# ============================================================


def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def taylor_var_increment_array(K, dt, TL):
    K = np.asarray(K, dtype=float)
    TL = np.asarray(TL, dtype=float)
    TL = np.maximum(TL, 1.0e-6)
    dt = float(dt)

    sigma2 = K / TL
    x = dt / TL

    return 2.0 * sigma2 * (TL**2) * (x - 1.0 + np.exp(-x))


def write_surfer_ascii_grd(filename, x_centers_m, y_centers_m, Z):
    zmin = float(np.min(Z))
    zmax = float(np.max(Z))

    with open(filename, "w") as f:
        f.write("DSAA\n")
        f.write(f"{len(x_centers_m)} {len(y_centers_m)}\n")
        f.write(f"{x_centers_m[0]:.6f} {x_centers_m[-1]:.6f}\n")
        f.write(f"{y_centers_m[0]:.6f} {y_centers_m[-1]:.6f}\n")
        f.write(f"{zmin:.12e} {zmax:.12e}\n")

        for j in range(len(y_centers_m)):
            row = Z[:, j]
            f.write(" ".join(f"{val:.12e}" for val in row) + "\n")


def wind_direction_from_uv(u, v):
    """
    Meteorological wind direction: direction FROM which wind blows.
    """
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def classify_stability_from_Lmo(Lmo):
    """
    Stability classification from Monin-Obukhov length.
    """
    Lmo = float(Lmo)

    if Lmo < 0.0:
        return "unstable"
    elif 0.0 < Lmo < 500.0:
        return "stable"
    elif Lmo >= 500.0:
        return "neutral"
    else:
        return "neutral"


def sigma_v_from_micromet(ustar, wstar, stab):
    ustar = np.asarray(ustar, dtype=float)
    wstar = np.asarray(wstar, dtype=float)

    ustar = np.maximum(ustar, 0.0)
    wstar = np.maximum(wstar, 0.0)

    sigma_vm2 = 3.6 * ustar**2

    if stab == "unstable":
        sigma_vc2 = 0.35 * wstar**2
    else:
        sigma_vc2 = 0.0

    sigv = np.sqrt(sigma_vm2 + sigma_vc2)
    sigv = np.clip(sigv, SIGV_MIN, SIGV_MAX)

    if sigv.ndim == 0:
        return float(sigv)

    return sigv


def sigma_w_from_micromet(ustar, wstar, stab):
    ustar = np.asarray(ustar, dtype=float)
    wstar = np.asarray(wstar, dtype=float)

    ustar = np.maximum(ustar, 0.0)
    wstar = np.maximum(wstar, 0.0)

    # Mechanical contribution
    sigma_wm2 = (1.3 * ustar) ** 2

    # Convective contribution
    if stab == "unstable":
        sigma_wc2 = 0.35 * wstar**2
    else:
        sigma_wc2 = 0.0

    sigw = np.sqrt(sigma_wm2 + sigma_wc2)

    return sigw


# ============================================================
# Puff arrays
# ============================================================
xp = np.empty(0, dtype=float)
yp = np.empty(0, dtype=float)
zp = np.empty(0, dtype=float)
sx = np.empty(0, dtype=float)
sy = np.empty(0, dtype=float)
sz = np.empty(0, dtype=float)
m = np.empty(0, dtype=float)  # puff mass in ug

# ============================================================
# Output accumulation field (ug/m3)
# ============================================================
C_sum_all_hours = np.zeros((nx, ny), dtype=float)

# ============================================================
# Main simulation loop
# ============================================================
t0 = time.perf_counter()
cpu0 = time.process_time()

for h in range(HOURS_RUN):
    u = float(u_day[h])
    v = float(v_day[h])
    Ta = float(Ta_day[h])
    Kz = float(Kz_day[h])
    Q_g_s = float(Q_g_s_day[h])
    Q_ug_s = Q_g_s * 1.0e6

    zi = float(zi_day[h])
    zi = max(50.0, zi)

    # Hourly gridded micromet fields for this absolute hour
    abs_h = START_HOUR + h
    ustar_grid_h = ustar_grid_full[abs_h]
    wstar_grid_h = wstar_grid_full[abs_h]

    Umean = float(np.sqrt(u * u + v * v))
    wd_from = wind_direction_from_uv(u, v)

    # Domain/station micromet from micromet_fluxes.csv, used for base scalar values
    ustar = float(ustar_day[h])
    wstar = float(wstar_day[h])
    Lmo = float(Lmo_day[h])

    stab = classify_stability_from_Lmo(Lmo)

    sigv = sigma_v_from_micromet(ustar, wstar, stab)
    sigv = clip(sigv, SIGV_MIN, SIGV_MAX)

    # plume rise -> effective height
    U_eff = max(Umean, 3.0 * ustar)
    dh = plume_rise_briggs(U_eff, Ta, H, D_stack, vs, Ts_K, stab, p=p_wind)

    H_eff = H + dh
    if H_eff >= zi:
        H_eff = zi

    # THIS MUST BE OUTSIDE the H_eff >= zi condition.
    C_sum = np.zeros((nx, ny), dtype=float)

    # Defaults for print in case all puffs disappear in first substep
    sea_frac_puffs = 0.0
    inside_frac_lu = 0.0

    for _ in range(substeps):
        # Spawn new puff
        xp = np.append(xp, x_src)
        yp = np.append(yp, y_src)
        zp = np.append(zp, H_eff)
        sx = np.append(sx, SIGMA0_H)
        sy = np.append(sy, SIGMA0_H)
        sz = np.append(sz, SIGMA0_Z)
        m = np.append(m, Q_ug_s * dt_sub)

        # Keep last max_puffs
        if xp.size > max_puffs:
            keep = slice(xp.size - max_puffs, xp.size)
            xp, yp, zp, sx, sy, sz, m = (
                xp[keep],
                yp[keep],
                zp[keep],
                sx[keep],
                sy[keep],
                sz[keep],
                m[keep],
            )

        # Advect
        xp = xp + u * dt_sub
        yp = yp + v * dt_sub

        # Land-use at puff positions
        lu_puff, inside_lu = sample_grid_at_points(
            xp, yp, landuse_grid, xmin, ymin, dx, dy
        )

        # Micromet at puff positions. sample_grid_at_points uses fallback values
        # for puffs outside the 20x20 grid, so we do not create NaNs in diffusion.
        ustar_puff, _ = sample_grid_at_points(xp, yp, ustar_grid_h, xmin, ymin, dx, dy)
        wstar_puff, _ = sample_grid_at_points(xp, yp, wstar_grid_h, xmin, ymin, dx, dy)

        sigv_puff = sigma_v_from_micromet(ustar_puff, wstar_puff, stab)
        # overwater safeguard
        sigv_puff = np.where(
            lu_puff == SEA_CLASS,
            np.maximum(sigv_puff, 1.5 * sigv),
            sigv_puff,
        )
        sigv_puff = np.clip(sigv_puff, SIGV_MIN, SIGV_MAX)

        sigw_puff = sigma_w_from_micromet(ustar_puff, wstar_puff, stab)

        sigw_puff = np.clip(sigw_puff, SIGW_MIN, SIGW_MAX)
        # Land-use Kxy factor
        Kxy_factor_puff = landuse_to_Kxy_factor(lu_puff)

        # --------------------------------------------------------
        # Wind-aligned horizontal dispersion
        # --------------------------------------------------------

        Kxy_puff = np.clip(sigv_puff * zi * Kxy_factor_puff, KXY_MIN, KXY_MAX)

        # Unit vector along wind direction
        U_safe = max(1.0e-6, np.sqrt(u * u + v * v))

        ex = u / U_safe
        ey = v / U_safe

        # Unit vector crosswind
        cx = -ey
        cy = ex

        # Along-wind and crosswind diffusivities
        # First controlled test: increase crosswind spreading
        K_along = 1.00 * Kxy_puff
        K_cross = 1.84 * Kxy_puff

        # Project wind-aligned diffusivity back to fixed x/y axes
        Kx = K_along * ex**2 + K_cross * cx**2
        Ky = K_along * ey**2 + K_cross * cy**2

        Kx = np.clip(Kx, KXY_MIN, KXY_MAX)
        Ky = np.clip(Ky, KXY_MIN, KXY_MAX)

        TLy_puff = Kxy_puff / np.maximum(sigv_puff**2, 1.0e-9)
        TLy_puff = np.clip(TLy_puff, 30.0, 3600.0)

        TLz_puff = Kz / np.maximum(sigw_puff**2, 1.0e-9)
        TLz_puff = np.clip(TLz_puff, 30.0, 3600.0)

        # Full Taylor variance growth
        dvar_x = taylor_var_increment_array(Kx, dt_sub, TLy_puff)
        dvar_y = taylor_var_increment_array(Ky, dt_sub, TLy_puff)
        dvar_z = taylor_var_increment_array(Kz, dt_sub, TLz_puff)

        sx = np.sqrt(sx * sx + dvar_x)
        sy = np.sqrt(sy * sy + dvar_y)
        sz = np.sqrt(sz * sz + dvar_z)

        # Puff splitting
        xp, yp, zp, sx, sy, sz, m = split_puffs_crosswind(
            xp,
            yp,
            zp,
            sx,
            sy,
            sz,
            m,
            u=u,
            v=v,
            sig_split=SIG_SPLIT,
            frac=SPLIT_OFFSET_FRAC,
        )

        # Guard: avoid puff explosion
        if xp.size > MAX_PUFFS_AFTER_SPLIT:
            keep = slice(xp.size - MAX_PUFFS_AFTER_SPLIT, xp.size)
            xp, yp, zp, sx, sy, sz, m = (
                xp[keep],
                yp[keep],
                zp[keep],
                sx[keep],
                sy[keep],
                sz[keep],
                m[keep],
            )

        # Drop puffs outside the buffered modeling area or too diffuse.
        # Keep puffs inside a buffered modeling area.
        in_box = (
            (xp >= xmin - buffer_m)
            & (xp <= xmax + buffer_m)
            & (yp >= ymin - buffer_m)
            & (yp <= ymax + buffer_m)
            & (sx <= sig_cap)
            & (sy <= sig_cap)
            & (sz <= sig_cap)
        )
        xp, yp, zp, sx, sy, sz, m = (
            xp[in_box],
            yp[in_box],
            zp[in_box],
            sx[in_box],
            sy[in_box],
            sz[in_box],
            m[in_box],
        )
        Umean = np.sqrt(u * u + v * v)
        sigma_adv = Umean * dt_sub / np.sqrt(12.0)
        sx_eff = np.sqrt(sx * sx + sigma_adv * sigma_adv)
        # Concentration at this substep (ug/m3)
        C_sum += compute_C_grid_with_zi(
            X, Y, z_rec, zi, xp, yp, zp, sx_eff, sy, sz, m, block=64, n_images=1
        )

    # hourly mean concentration (ug/m3)
    C_hour_mean = C_sum / substeps

    # Accumulate results only for the requested output period.
    # Spin-up hours affect puff build-up, but are not included in final averaging.
    abs_h = START_HOUR + h

    if OUTPUT_START_HOUR <= abs_h < OUTPUT_END_HOUR:
        C_sum_all_hours += C_hour_mean

    if h % 6 == 0:
        completed_output_hours = max(
            1,
            min(abs_h + 1, OUTPUT_END_HOUR) - OUTPUT_START_HOUR,
        )
        C_running_mean = C_sum_all_hours / completed_output_hours
        print(
            f"Hour {h+1}/{HOURS_RUN} | "
            f"puffs={xp.size} | "
            f"max={C_running_mean.max():.3e} ug/m3 | "
            f"Q={Q_g_s:.2f} g/s | "
            f"stab={stab} | "
        )

# ============================================================
# Final mean concentration field (ug/m3)
# ============================================================
C_mean_all = C_sum_all_hours / HOURS_OUTPUT

# ============================================================
# Statistics: max mean concentration and receptor location
# ============================================================
imax_flat = np.argmax(C_mean_all)
ix0, iy0 = np.unravel_index(imax_flat, C_mean_all.shape)

ix = ix0 + 1
iy = iy0 + 1

x_max = X[ix0, iy0]
y_max = Y[ix0, iy0]
c_max_mean = C_mean_all[ix0, iy0]

# ============================================================
# EXPORT FILE NAMES
# ============================================================
out_tag = PERIOD_TAG

txt_name = RESULTS_DIR / f"CONCENTRATIONS_{out_tag}.txt"
grd_name = RESULTS_DIR / f"CONCENTRATIONS_{out_tag}.grd"

# Export TXT
# ============================================================
rows = []
for i in range(nx):
    for j in range(ny):
        rows.append([i + 1, j + 1, X[i, j], Y[i, j], C_mean_all[i, j]])
rows = np.array(rows, dtype=float)

if SAVE_TXT:
    np.savetxt(
        txt_name,
        rows,
        fmt=["%d", "%d", "%.6f", "%.6f", "%.12e"],
        header="ix iy utm_x_m utm_y_m C_mean_ug_per_m3",
        comments="",
    )

# ============================================================
# Export GRD raster file (meters, ug/m3)
# ============================================================
if SAVE_GRD:
    write_surfer_ascii_grd(grd_name, x, y, C_mean_all)

# ============================================================
# Final statistics
# ============================================================
print("\n===== FINAL STATISTICS =====")
print("Average concentration field (ug/m3):")
print("min  =", float(C_mean_all.min()))
print("max  =", float(C_mean_all.max()))
print("mean =", float(C_mean_all.mean()))
print(
    f"Max mean concentration observed at receptor ({ix},{iy}) "
    f"with coordinates x={x_max:.3f} m, y={y_max:.3f} m"
)

elapsed = time.perf_counter() - t0
cpu_elapsed = time.process_time() - cpu0

print("\n===== COMPUTATIONAL COST =====")
print(f"Total wall-clock runtime : {elapsed:.2f} s")
print(f"CPU processing time : {cpu_elapsed:.2f} s")

if SAVE_TXT:
    print(f"OK: wrote {txt_name}")
if SAVE_GRD:
    print(f"OK: wrote {grd_name}")

# ============================================================
# Optional contour plot (lines only, no fill)
# ============================================================
if SHOW_PLOT:
    plt.figure(figsize=(8, 7))
    cs = plt.contour(X, Y, C_mean_all, levels=10, linewidths=1.0)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.1e")

    plt.scatter(x_src, y_src, marker="*", s=120, label="Source")
    plt.scatter(x_max, y_max, marker="o", s=60, label=f"Max mean receptor ({ix},{iy})")

    plt.xlabel("UTM x (m)")
    plt.ylabel("UTM y (m)")
    plt.title(f"Isoconcentration contours of mean concentration ({out_tag})")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
