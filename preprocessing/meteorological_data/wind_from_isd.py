import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
YEAR = 2024

# Repository root:
# atmospheric-dispersion-model/
# ├── input/surface/
# ├── processed/
# └── preprocessing/meteorological_data/wind_from_isd.py
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "input" / "surface"
OUTPUT_DIR = BASE_DIR / "processed"

# Change only the raw ISD filename when using another station/year.
INFILE = INPUT_DIR / "167161-99999-2024"
OUTFILE = OUTPUT_DIR / "wind_from_isd.csv"

START = datetime(YEAR, 1, 1, 0, 0, 0)
END = datetime(YEAR + 1, 1, 1, 0, 0, 0)

# Automatically gives 8760 h for a normal year and 8784 h for a leap year.
HOURS_YEAR = int((END - START).total_seconds() // 3600)


def met_wind_to_uv(ws, wd_deg):
    # direction is from, so the minus sign
    th = np.deg2rad(wd_deg)
    u = -ws * np.sin(th)
    v = -ws * np.cos(th)
    return u, v


def main():
    if not INFILE.exists():
        raise FileNotFoundError(
            f"ISD input file not found:\n{INFILE}\n"
            "Place the raw NOAA ISD file in input/surface/ "
            "and update INFILE in USER SETTINGS."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing NOAA ISD wind observations for {YEAR}")
    print(f"Input : {INFILE}")
    print(f"Output: {OUTFILE}")
    print(f"Expected hours in year: {HOURS_YEAR}")

    rows = []

    with open(INFILE, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if len(ln) < 93:
                continue

            try:
                # fixed width columns from NOAA
                datestr = ln[15:23]  # year-month-day
                timestr = ln[23:27]  # hour-minute

                wd_str = ln[60:63]  # wind direction deg
                wd_qc = ln[63]  # quality code
                ws_str = ln[65:69]  # wind speed  1/10m/s
                ws_qc = ln[69]  # quality code

                tmp_str = ln[87:92]  # air temp,  1/10C
                tmp_qc = ln[92]  # quality code
            except Exception:
                continue

            # missing data
            if (
                wd_str == "999"
                or ws_str == "9999"
                or tmp_str == "+9999"
                or tmp_str == "-9999"
            ):
                continue

            try:
                year = int(datestr[0:4])
                month = int(datestr[4:6])
                day = int(datestr[6:8])

                hour = int(timestr[0:2])
                minute = int(timestr[2:4])

                wd = float(wd_str)
                ws = float(ws_str) / 10.0  # m/s
                Ta_C = float(tmp_str) / 10.0  # deg C
                Ta_K = Ta_C + 273.15
            except ValueError:
                continue

            if year != YEAR:
                continue

            try:
                t = datetime(year, month, day, hour, minute)
            except ValueError:
                continue

            t_hour = datetime(year, month, day, hour, 0, 0)

            hour_index = int((t_hour - START).total_seconds() // 3600)
            if not (0 <= hour_index < HOURS_YEAR):
                continue

            u, v = met_wind_to_uv(ws, wd)
            rows.append((hour_index, u, v, Ta_K))

    if not rows:
        raise RuntimeError("No valid rows from NOAA file.")

    tmp = np.array(rows, dtype=float)

    h = tmp[:, 0].astype(int)
    u = tmp[:, 1]
    v = tmp[:, 2]
    Ta = tmp[:, 3]

    uniq = np.unique(h)
    u_mean = np.zeros_like(uniq, dtype=float)
    v_mean = np.zeros_like(uniq, dtype=float)
    Ta_mean = np.zeros_like(uniq, dtype=float)

    for i, hh in enumerate(uniq):
        mask = h == hh
        u_mean[i] = u[mask].mean()
        v_mean[i] = v[mask].mean()
        Ta_mean[i] = Ta[mask].mean()

    out = np.column_stack([uniq, u_mean, v_mean, Ta_mean])

    np.savetxt(
        OUTFILE,
        out,
        delimiter=",",
        header="hour_index,u_mps,v_mps,Ta_K",
        comments="",
        fmt="%.6f",
    )

    print(f"OK wrote {OUTFILE} with {out.shape[0]} hours.")
    print("hour_index range:", int(out[0, 0]), "to", int(out[-1, 0]))
    # print("First 5 rows:\n", out[:5])


if __name__ == "__main__":
    main()
