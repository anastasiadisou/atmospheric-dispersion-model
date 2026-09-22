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
# └── preprocessing/meteorological_data/kz_from_igra.py
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "input" / "upper_air"
OUTPUT_DIR = BASE_DIR / "processed"

# Change only the raw IGRA filename when using another station/year.
IGRA_FILE = INPUT_DIR / "GRM00016716_2024.txt"
OUTPUT_CSV = OUTPUT_DIR / "kz.csv"

# Kz parameterization
KZ_MIN = 3.0
KZ_MAX = 80.0
DV_REF = 20.0  # m/s (reference wind-speed difference)

# Pressure levels (Pa)
P_LOW_TARGETS = [100000, 92500]
P_HIGH_TARGETS = [85000]

START_DT = datetime(YEAR, 1, 1, 0)
END_DT = datetime(YEAR + 1, 1, 1, 0)

# Automatically gives 8760 h for a normal year and 8784 h for a leap year.
HOURS_YEAR = int((END_DT - START_DT).total_seconds() // 3600)



def safe_int(s: str):
    s = s.strip()
    if s == "" or s == "-9999":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_header(line: str):

    parts = line.strip().split()
    if len(parts) < 7:
        return None
    if not parts[0].startswith("#"):
        return None

    station_id = parts[0][1:]
    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    hour = int(parts[4])
    numlev = int(parts[6])
    return station_id, year, month, day, hour, numlev


def parse_level_fixedwidth(line: str):
    # using IGRA fixed columns
    s = line.rstrip("\n")
    if len(s) < 51:
        s = s.ljust(51)

    press = safe_int(s[9:15])
    gph = safe_int(s[16:21])
    wdir = safe_int(s[40:45])
    wspd = safe_int(s[46:51])

    if press is None or wdir is None or wspd is None:
        return None

    ws = wspd / 10.0  # m/s
    wd = wdir  # deg

    if ws < 0 or wd < 0:
        return None

    return {"press": press, "gph": gph, "ws": ws, "wd": wd}


def nearest_level(levels, targets):
    # find the pressures closer to the targets
    if not levels:
        return None

    presses = np.array([lv["press"] for lv in levels], dtype=float)
    best = None
    best_dist = 1e18

    for t in targets:
        dist = np.abs(presses - t)
        idx = int(np.argmin(dist))
        if dist[idx] < best_dist:
            best_dist = dist[idx]
            best = levels[idx]

    return best


def kz_from_dv(dv: float) -> float:
    frac = np.clip(dv / DV_REF, 0.0, 1.0)
    return KZ_MIN + (KZ_MAX - KZ_MIN) * frac


def main():
    if not IGRA_FILE.exists():
        raise FileNotFoundError(
            f"IGRA input file not found:\n{IGRA_FILE}\n"
            "Place the raw IGRA file in input/upper_air/ "
            "and update IGRA_FILE in USER SETTINGS."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing IGRA upper-air observations for {YEAR}")
    print(f"Input : {IGRA_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Expected hourly rows: {HOURS_YEAR}")

    # compute Kz
    times = []
    kz_vals = []

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

        # Skip other years
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

        low = nearest_level(levels, P_LOW_TARGETS)
        high = nearest_level(levels, P_HIGH_TARGETS)
        if low is None or high is None:
            continue

        dv = abs(high["ws"] - low["ws"])
        kz = kz_from_dv(dv)

        times.append(datetime(year, month, day, hour))
        kz_vals.append(kz)

    times = np.array(times, dtype="datetime64[h]")
    kz_vals = np.array(kz_vals, dtype=float)

    if kz_vals.size == 0:
        raise RuntimeError("Parsed soundings: 0.")

    print("Parsed soundings:", len(kz_vals))
    print(
        "Kz stats:",
        "min",
        float(kz_vals.min()),
        "max",
        float(kz_vals.max()),
        "mean",
        float(kz_vals.mean()),
    )

    # interpolation
    t0 = np.datetime64(START_DT, "h")

    x_hours = np.arange(HOURS_YEAR, dtype=int)
    snd_hours = (times - t0).astype(int)

    ok = (snd_hours >= 0) & (snd_hours < HOURS_YEAR)
    snd_hours = snd_hours[ok]
    kz_vals = kz_vals[ok]

    order = np.argsort(snd_hours)
    snd_hours = snd_hours[order]
    kz_vals = kz_vals[order]

    if len(kz_vals) < 2:
        kz_hourly = np.full(
            HOURS_YEAR,
            float(np.nanmean(kz_vals)) if len(kz_vals) else (KZ_MIN + KZ_MAX) / 2,
        )
    else:
        kz_hourly = np.interp(x_hours, snd_hours, kz_vals)

    #  kz.csv
    out = np.column_stack([x_hours, kz_hourly])
    np.savetxt(
        OUTPUT_CSV,
        out,
        delimiter=",",
        header="hour_index,Kz_m2ps",
        comments="",
        fmt="%.6f",
    )
    print(f"OK wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
