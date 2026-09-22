import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
YEAR = 2024

# Repository root:
# atmospheric-dispersion-model/
# ├── input/upper_air/
# ├── processed/
# └── preprocessing/meteorological_data/zi_from_igra.py
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "input" / "upper_air"
OUTPUT_DIR = BASE_DIR / "processed"

# Change only the raw IGRA filename when using another station/year.
IGRA_FILE = INPUT_DIR / "GRM00016716_2024.txt"
OUTPUT_CSV = OUTPUT_DIR / "zi.csv"

# Search limits for mixing height zi (m)
MIN_Z_M = 100.0
MAX_Z_M = 3000.0

# Strongest-inversion method parameters
SMOOTH_WINDOW = 3
MIN_LEVELS_REQUIRED = 8
THETA_GRAD_MIN = 0.002  # K/m

START_DT = datetime(YEAR, 1, 1, 0)
END_DT = datetime(YEAR + 1, 1, 1, 0)

# Automatically gives 8760 h for a normal year and 8784 h for a leap year.
HOURS_YEAR = int((END_DT - START_DT).total_seconds() // 3600)



def clean_int_from_field(field: str):
    s = field.strip()
    if not s:
        return None
    out = []
    i = 0
    if i < len(s) and s[i] in "+-":
        out.append(s[i])
        i += 1
    while i < len(s) and s[i].isdigit():
        out.append(s[i])
        i += 1
    if not out or out == ["+"] or out == ["-"]:
        return None
    try:
        val = int("".join(out))
    except ValueError:
        return None
    if val == -9999:
        return None
    return val


def parse_header(line: str):
    parts = line.strip().split()
    if len(parts) < 7 or not parts[0].startswith("#"):
        return None
    station_id = parts[0][1:]
    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    hour = int(parts[4])
    numlev = int(parts[6])
    return station_id, year, month, day, hour, numlev


def parse_level_fixedwidth(line: str):
    # fixed width IGRA columns
    s = line.rstrip("\n")
    if len(s) < 27:
        s = s.ljust(27)

    press = clean_int_from_field(s[9:15])
    gph = clean_int_from_field(s[16:21])
    temp = clean_int_from_field(s[22:27])

    if press is None or gph is None or temp is None:
        return None

    return {"press": float(press), "z": float(gph), "T_K": float(temp / 10.0 + 273.15)}


def potential_temperature(T_K: np.ndarray, p_Pa: np.ndarray):
    p0 = 100000.0
    kappa = 0.2854
    return T_K * (p0 / p_Pa) ** kappa


def moving_average(x: np.ndarray, w: int):
    if w <= 1 or x.size < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def estimate_zi_strongest_inversion(levels):
    # Strongest-inversion method - theta(z)-> dtheta/dz -> MAX gradient location ->  zi is where the strongest inversion layer is
    if len(levels) < MIN_LEVELS_REQUIRED:
        return None

    # sort by z
    levels = sorted(levels, key=lambda d: d["z"])
    z = np.array([d["z"] for d in levels], dtype=float)
    p = np.array([d["press"] for d in levels], dtype=float)
    T = np.array([d["T_K"] for d in levels], dtype=float)

    ok = np.isfinite(z) & np.isfinite(p) & np.isfinite(T)
    z, p, T = z[ok], p[ok], T[ok]
    if z.size < MIN_LEVELS_REQUIRED:
        return None

    # ensure strictly increasing z
    order = np.argsort(z)
    z, p, T = z[order], p[order], T[order]
    uniq = np.ones_like(z, dtype=bool)
    uniq[1:] = z[1:] > z[:-1]
    z, p, T = z[uniq], p[uniq], T[uniq]
    if z.size < MIN_LEVELS_REQUIRED:
        return None

    theta = potential_temperature(T, p)

    dz = np.diff(z)
    dth = np.diff(theta)
    good = dz > 1.0
    if np.count_nonzero(good) < 3:
        return None

    # gradient defined at "upper" point i+1
    grad = np.full(z.size, np.nan, dtype=float)
    grad_mid = dth[good] / dz[good]
    idx = np.where(good)[0]
    grad[idx + 1] = grad_mid

    grad = moving_average(grad, SMOOTH_WINDOW)

    region = (z >= MIN_Z_M) & (z <= MAX_Z_M) & np.isfinite(grad)
    if not np.any(region):
        return None

    # strongest inversion = max grad
    j = np.where(region)[0]
    jmax = j[np.nanargmax(grad[j])]

    if not np.isfinite(grad[jmax]) or grad[jmax] < THETA_GRAD_MIN:
        return None

    # base of inversion layer
    jbase = max(0, jmax - 1)
    zi = float(z[jbase])

    return zi


def main():
    if not IGRA_FILE.exists():
        raise FileNotFoundError(
            f"IGRA input file not found:\n{IGRA_FILE}\n"
            "Place the raw IGRA file in input/upper_air/ "
            "and update IGRA_FILE in USER SETTINGS."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing IGRA mixing-height observations for {YEAR}")
    print(f"Input : {IGRA_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Expected hourly rows: {HOURS_YEAR}")

    # reading igra
    times = []
    zi_vals = []

    with open(IGRA_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("#"):
            i += 1
            continue

        hdr = parse_header(line)
        if hdr is None:
            i += 1
            continue

        station_id, year, month, day, hour, numlev = hdr
        i += 1

        if year != YEAR:
            i += numlev
            continue

        levels = []
        for _ in range(numlev):
            if i >= len(lines):
                break
            lv = parse_level_fixedwidth(lines[i])
            if lv is not None:
                levels.append(lv)
            i += 1

        zi = estimate_zi_strongest_inversion(levels)
        zi_vals.append(np.nan if zi is None else float(zi))
        times.append(datetime(year, month, day, hour))

    times64 = np.array(times, dtype="datetime64[h]")
    zi_vals = np.array(zi_vals, dtype=float)

    if zi_vals.size == 0:
        raise RuntimeError("Parsed soundings: 0. Check IGRA.")

    valid = np.isfinite(zi_vals)
    print("Parsed soundings:", len(zi_vals))
    print("Valid zi soundings:", int(valid.sum()), "Missing zi:", int((~valid).sum()))

    # interpolation
    t0 = np.datetime64(START_DT, "h")
    x_hours = np.arange(HOURS_YEAR, dtype=int)

    snd_hours = (times64 - t0).astype(int)
    inside = (snd_hours >= 0) & (snd_hours < HOURS_YEAR)

    snd_hours = snd_hours[inside]
    zi_vals = zi_vals[inside]

    valid = np.isfinite(zi_vals)
    snd_h = snd_hours[valid]
    zi_v = zi_vals[valid]

    if zi_v.size == 0:
        raise RuntimeError("No valid zi values detected.")
    elif zi_v.size == 1:
        zi_hourly = np.full(HOURS_YEAR, float(zi_v[0]))
    else:
        order = np.argsort(snd_h)
        snd_h = snd_h[order]
        zi_v = zi_v[order]
        zi_hourly = np.interp(x_hours, snd_h, zi_v)

    # forward/back fill
    if not np.isfinite(zi_hourly[0]):
        first = np.argmax(np.isfinite(zi_hourly))
        zi_hourly[:first] = zi_hourly[first]
    if not np.isfinite(zi_hourly[-1]):
        last = len(zi_hourly) - 1 - np.argmax(np.isfinite(zi_hourly[::-1]))
        zi_hourly[last + 1 :] = zi_hourly[last]

    # final fill
    for k in range(HOURS_YEAR):
        if not np.isfinite(zi_hourly[k]) and k > 0:
            zi_hourly[k] = zi_hourly[k - 1]
    for k in range(HOURS_YEAR - 1, -1, -1):
        if not np.isfinite(zi_hourly[k]) and k < HOURS_YEAR - 1:
            zi_hourly[k] = zi_hourly[k + 1]

    out = np.column_stack([x_hours, zi_hourly])
    np.savetxt(
        OUTPUT_CSV,
        out,
        delimiter=",",
        header="hour_index,zi_m",
        comments="",
        fmt="%.6f",
    )

    print(f"OK wrote {OUTPUT_CSV}")
    print(
        "zi stats:",
        "min",
        float(np.min(zi_hourly)),
        "max",
        float(np.max(zi_hourly)),
        "mean",
        float(np.mean(zi_hourly)),
    )


if __name__ == "__main__":
    main()
