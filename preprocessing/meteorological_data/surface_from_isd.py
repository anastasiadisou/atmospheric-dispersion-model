import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
YEAR = 2024

# Repository root:
# atmospheric-dispersion-model/
# ├── input/surface/
# ├── processed/
# └── preprocessing/meteorological_data/surface_from_isd.py
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "input" / "surface"
OUTPUT_DIR = BASE_DIR / "processed"

# Change only the raw ISD filename when using another station/year.
ISD_INPUT_FILE = INPUT_DIR / "167161-99999-2024"
OUTPUT_CSV = OUTPUT_DIR / "surface_from_isd.csv"

START_DT = datetime(YEAR, 1, 1, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(YEAR + 1, 1, 1, 0, 0, tzinfo=timezone.utc)

# Automatically gives 8760 h for a normal year and 8784 h for a leap year.
HOURS_YEAR = int((END_DT - START_DT).total_seconds() // 3600)


# ============================================================
# Helpers
# ============================================================
def safe_int(s, default=None):
    try:
        return int(s)
    except Exception:
        return default


def parse_scaled_signed_int(s, missing_value, scale=1.0):
    s = s.strip()
    if s == missing_value:
        return None
    try:
        return int(s) / scale
    except Exception:
        return None


def parse_scaled_unsigned_int(s, missing_value, scale=1.0):
    s = s.strip()
    if s == missing_value:
        return None
    try:
        return int(s) / scale
    except Exception:
        return None


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def parse_metar_qnh_hpa(line):
    """
    Fallback parser for pressure from METAR QNH token.

    Many ISD records may have the fixed sea-level pressure field as 99999,
    while the METAR remarks still contain pressure as Q1019, Q1018, etc.

    Returns:
        pressure in hPa, e.g. Q1019 -> 1019.0
    """
    m = re.search(r"\bQ(\d{4})\b", line)
    if not m:
        return None

    try:
        return float(m.group(1))
    except Exception:
        return None


# ============================================================
# Solar geometry for is_day
# Simple, robust approximation
# ============================================================
def day_of_year(dt):
    return dt.timetuple().tm_yday


def solar_declination_rad(n):
    # Cooper approximation
    return math.radians(23.45) * math.sin(math.radians(360.0 * (284 + n) / 365.0))


def equation_of_time_minutes(n):
    # Approximation
    B = math.radians(360.0 * (n - 81) / 364.0)
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def solar_zenith_deg(dt_utc, lat_deg, lon_deg):
    n = day_of_year(dt_utc)
    dec = solar_declination_rad(n)
    lat = math.radians(lat_deg)

    eot = equation_of_time_minutes(n)

    # UTC decimal hours
    utc_hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # local solar time
    lst_hours = utc_hours + lon_deg / 15.0 + eot / 60.0

    hra = math.radians(15.0 * (lst_hours - 12.0))

    cos_zen = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(
        hra
    )
    cos_zen = clip(cos_zen, -1.0, 1.0)
    zen = math.degrees(math.acos(cos_zen))
    return zen


def is_daytime(dt_utc, lat_deg, lon_deg):
    zen = solar_zenith_deg(dt_utc, lat_deg, lon_deg)
    return 1 if zen < 90.0 else 0


# ============================================================
# METAR cloud parsing fallback
# ============================================================
METAR_CLOUD_MAP = {
    "SKC": 0,
    "CLR": 0,
    "NSC": 0,
    "NCD": 0,
    "FEW": 2,
    "SCT": 4,
    "BKN": 6,
    "OVC": 8,
    "VV": 8,  # vertical visibility / obscured -> treat as overcast-ish
}


def cloud_from_metar_text(line):
    """
    Fallback parser from raw METAR remarks text.
    Returns:
      oktas, ceiling_m
    """
    oktas = None
    ceiling_m = None

    # Look inside whole line; if METAR exists, fine.
    # Find cloud tokens like FEW020, SCT025, BKN030, OVC050, VV003
    matches = re.findall(r"\b(FEW|SCT|BKN|OVC|VV|SKC|CLR|NSC|NCD)(\d{3})?\b", line)

    if not matches:
        return None, None

    all_oktas = []
    ceiling_candidates = []

    for code, hhh in matches:
        if code in METAR_CLOUD_MAP:
            all_oktas.append(METAR_CLOUD_MAP[code])

        # cloud base in hundreds of feet -> meters
        if hhh and code in ("FEW", "SCT", "BKN", "OVC", "VV"):
            try:
                h_ft = int(hhh) * 100.0
                h_m = h_ft * 0.3048
                ceiling_candidates.append(h_m)
            except Exception:
                pass

    if all_oktas:
        oktas = max(all_oktas)

    # Ceiling typically BKN/OVC/VV, but if only FEW/SCT exist, keep lowest cloud base as fallback
    strict_ceiling = []
    for code, hhh in matches:
        if hhh and code in ("BKN", "OVC", "VV"):
            strict_ceiling.append(int(hhh) * 100.0 * 0.3048)

    if strict_ceiling:
        ceiling_m = min(strict_ceiling)
    elif ceiling_candidates:
        ceiling_m = min(ceiling_candidates)

    return oktas, ceiling_m


# ============================================================
# ISD additional groups parsing
# ============================================================
def parse_gf1_total_coverage(additional_text):
    """
    Parse GF1 group if present.
    Returns:
      oktas (0-8) or None
    """
    m = re.search(r"GF1(\d{2})(\d{2})([0-9A-Z])", additional_text)
    if not m:
        return None

    total_cov_code = m.group(1)

    # Map total coverage code to oktas
    mapping = {
        "00": 0,
        "01": 1,
        "02": 2,
        "03": 3,
        "04": 4,
        "05": 5,
        "06": 6,
        "07": 7,
        "08": 8,
        "09": 8,  # obscured -> high cover
        "10": 8,  # partial obscuration -> treat conservatively
        "11": 2,
        "12": 4,
        "13": 4,
        "14": 6,
        "15": 6,
        "16": 6,
        "17": 8,
        "18": 8,
        "19": 8,
    }
    return mapping.get(total_cov_code, None)


def parse_ga_layers(additional_text):
    """
    Parse up to 6 GA groups.
    Returns:
      max_oktas, min_base_m
    """
    # GA1-GA6: GAx + coverage code(2) + cov qc(1) + base height(6) + base qc(1) + cloud type(2) + type qc(1)
    pattern = r"GA[1-6](\d{2})([0-9A-Z])([+\-]\d{5})([0-9A-Z])(\d{2})([0-9A-Z])"
    matches = re.findall(pattern, additional_text)

    if not matches:
        return None, None

    oktas_list = []
    base_list = []

    cov_map = {
        "00": 0,
        "01": 1,
        "02": 2,
        "03": 3,
        "04": 4,
        "05": 5,
        "06": 6,
        "07": 7,
        "08": 8,
        "09": 8,
        "10": 8,
    }

    for cov_code, cov_qc, base_str, base_qc, cloud_type, type_qc in matches:
        if cov_code in cov_map:
            oktas_list.append(cov_map[cov_code])

        if base_str != "+99999":
            try:
                base_list.append(int(base_str))
            except Exception:
                pass

    oktas = max(oktas_list) if oktas_list else None
    min_base = min(base_list) if base_list else None
    return oktas, min_base


def parse_gd_summation(additional_text):
    """
    Parse GD1-GD6 sky-cover-summation as fallback.
    Returns:
      max_oktas, lowest_height_m
    """
    # GDx + cov1(1) + cov2(2) + qc(1) + height(6) + height_qc(1) + characteristic(1)
    pattern = r"GD[1-6]([0-9A-Z])(\d{2})([0-9A-Z])([+\-]\d{5})([0-9A-Z])([0-9A-Z])"
    matches = re.findall(pattern, additional_text)

    if not matches:
        return None, None

    oktas_list = []
    height_list = []

    cov1_map = {
        "0": 0,
        "1": 2,  # FEW
        "2": 4,  # SCT
        "3": 6,  # BKN
        "4": 8,  # OVC
        "5": 8,
        "6": 8,
    }

    cov2_map = {
        "00": 0,
        "01": 1,
        "02": 2,
        "03": 3,
        "04": 4,
        "05": 5,
        "06": 6,
        "07": 7,
        "08": 8,
        "09": 8,
        "10": 8,
        "11": 2,
        "12": 4,
        "13": 4,
        "14": 6,
        "15": 6,
        "16": 6,
        "17": 8,
        "18": 8,
        "19": 8,
    }

    for cov1, cov2, qc, h_str, h_qc, char_code in matches:
        if cov2 in cov2_map:
            oktas_list.append(cov2_map[cov2])
        elif cov1 in cov1_map:
            oktas_list.append(cov1_map[cov1])

        if h_str != "+99999":
            try:
                height_list.append(int(h_str))
            except Exception:
                pass

    oktas = max(oktas_list) if oktas_list else None
    height = min(height_list) if height_list else None
    return oktas, height


def parse_gh1_solar(additional_text):
    """
    Parse GH1 hourly average solar radiation [W/m2]
    GH1 + SOLARAD(5) + QC(1) + FLAG(1) + ...
    scaling factor 10
    """
    m = re.search(r"GH1(\d{5})([0-9A-Z])([0-9A-Z])", additional_text)
    if not m:
        return None

    raw = m.group(1)
    if raw == "99999":
        return None

    try:
        return int(raw) / 10.0
    except Exception:
        return None


# ============================================================
# Main fixed mandatory parser
# ============================================================
def parse_isd_line(line):
    """
    Parses one raw ISD line.
    Requires line length >= 105 chars for fixed + mandatory sections.
    """
    line = line.rstrip("\n")

    if len(line) < 105:
        return None

    # Control section (1-based -> Python slices)
    date_str = line[15:23]  # positions 16-23
    time_str = line[23:27]  # positions 24-27
    lat_str = line[28:34]  # positions 29-34
    lon_str = line[34:41]  # positions 35-41
    report_type = line[41:46].strip()  # positions 42-46
    elev_str = line[46:51]  # positions 47-51

    # Mandatory section
    wd_str = line[60:63]  # 61-63
    wind_type = line[64:65]  # 65
    ws_str = line[65:69]  # 66-69
    ceiling_str = line[70:75]  # 71-75
    visibility_str = line[78:84]  # 79-84
    ta_str = line[87:92]  # 88-92
    td_str = line[93:98]  # 94-98
    slp_str = line[99:104]  # 100-104

    # Basic parse
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return None

    lat_deg = parse_scaled_signed_int(lat_str, "+99999", scale=1000.0)
    lon_deg = parse_scaled_signed_int(lon_str, "+999999", scale=1000.0)
    elev_m = parse_scaled_signed_int(elev_str, "+9999", scale=1.0)

    wd_deg = parse_scaled_unsigned_int(wd_str, "999", scale=1.0)
    u10_mps = parse_scaled_unsigned_int(ws_str, "9999", scale=10.0)
    ceiling_m = parse_scaled_unsigned_int(ceiling_str, "99999", scale=1.0)
    visibility_m = parse_scaled_unsigned_int(visibility_str, "999999", scale=1.0)

    ta_c = parse_scaled_signed_int(ta_str, "+9999", scale=10.0)
    td_c = parse_scaled_signed_int(td_str, "+9999", scale=10.0)
    slp_hpa = parse_scaled_unsigned_int(slp_str, "99999", scale=10.0)

    # Fallback from METAR QNH, e.g. Q1019 -> 1019 hPa.
    # This is needed for files where the fixed ISD SLP field is 99999,
    # but the actual pressure exists in the METAR part of the same line.
    if slp_hpa is None:
        slp_hpa = parse_metar_qnh_hpa(line)

    Ta_K = ta_c + 273.15 if ta_c is not None else None
    Td_K = td_c + 273.15 if td_c is not None else None

    # Additional section and remarks
    rest = line[105:] if len(line) > 105 else ""

    # Try structured cloud / solar parsing
    cloud_oktas = None
    solar_wm2 = None

    cloud_oktas_gf = parse_gf1_total_coverage(rest)
    cloud_oktas_ga, ga_base = parse_ga_layers(rest)
    cloud_oktas_gd, gd_height = parse_gd_summation(rest)
    solar_wm2 = parse_gh1_solar(rest)

    # Priority for total cloud:
    # GF1 > GD > GA > METAR remarks
    if cloud_oktas_gf is not None:
        cloud_oktas = cloud_oktas_gf
    elif cloud_oktas_gd is not None:
        cloud_oktas = cloud_oktas_gd
    elif cloud_oktas_ga is not None:
        cloud_oktas = cloud_oktas_ga

    # If ceiling missing, use layer heights as fallback
    if ceiling_m is None:
        fallback_base = None
        bases = [x for x in [ga_base, gd_height] if x is not None]
        if bases:
            fallback_base = min(bases)
        if fallback_base is not None:
            ceiling_m = fallback_base

    # METAR fallback
    if cloud_oktas is None or ceiling_m is None:
        metar_oktas, metar_ceiling = cloud_from_metar_text(line)
        if cloud_oktas is None:
            cloud_oktas = metar_oktas
        if ceiling_m is None:
            ceiling_m = metar_ceiling

    cloud_fraction = None
    if cloud_oktas is not None:
        cloud_fraction = cloud_oktas / 8.0

    # Day/night
    is_day = None
    if lat_deg is not None and lon_deg is not None:
        is_day = is_daytime(dt, lat_deg, lon_deg)

    return {
        "datetime_utc": dt,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "elev_m": elev_m,
        "report_type": report_type,
        "u10_mps": u10_mps,
        "wd_deg": wd_deg,
        "Ta_K": Ta_K,
        "Td_K": Td_K,
        "slp_hPa": slp_hpa,
        "ceiling_m": ceiling_m,
        "visibility_m": visibility_m,
        "cloud_total_oktas": cloud_oktas,
        "cloud_fraction": cloud_fraction,
        "solar_wm2": solar_wm2,
        "is_day": is_day,
        "raw_line": line,
    }


# ============================================================
# Hourly aggregation
# Keep the last valid record in each hour
# ============================================================
def hour_index_from_datetime(dt):
    return int((dt - START_DT).total_seconds() // 3600)


def merge_hour_record(existing, new):
    """
    Keep latest record in the hour, but backfill missing fields from earlier if needed.
    """
    if existing is None:
        return new

    merged = existing.copy()

    # If the new record is later in the hour, prefer its values when present
    for key, value in new.items():
        if key == "datetime_utc":
            if new["datetime_utc"] >= existing["datetime_utc"]:
                merged["datetime_utc"] = new["datetime_utc"]
        elif key == "raw_line":
            if new["datetime_utc"] >= existing["datetime_utc"]:
                merged["raw_line"] = new["raw_line"]
        else:
            if new["datetime_utc"] >= existing["datetime_utc"]:
                if value is not None:
                    merged[key] = value
            else:
                if merged.get(key) is None and value is not None:
                    merged[key] = value

    return merged


# ============================================================
# Main
# ============================================================
def main():
    if not ISD_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ISD input file not found:\n{ISD_INPUT_FILE}\n"
            "Place the raw NOAA ISD file in input/surface/ "
            "and update ISD_INPUT_FILE in USER SETTINGS."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = {}

    total_lines = 0
    parsed_records = 0
    fixed_slp_valid_count = 0
    metar_qnh_valid_count = 0

    print(f"Processing NOAA ISD surface observations for {YEAR}")
    print(f"Input : {ISD_INPUT_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Expected hourly rows: {HOURS_YEAR}")

    with open(ISD_INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            total_lines += 1

            # Simple pressure diagnostics for this raw line
            if len(line) >= 104:
                fixed_slp = parse_scaled_unsigned_int(line[99:104], "99999", scale=10.0)
                if fixed_slp is not None:
                    fixed_slp_valid_count += 1

            if parse_metar_qnh_hpa(line) is not None:
                metar_qnh_valid_count += 1

            rec = parse_isd_line(line)
            if rec is None:
                continue

            parsed_records += 1

            dt = rec["datetime_utc"]
            if dt < START_DT or dt >= END_DT:
                continue

            hidx = hour_index_from_datetime(dt)
            if hidx < 0 or hidx >= HOURS_YEAR:
                continue

            if hidx not in hourly:
                hourly[hidx] = rec
            else:
                hourly[hidx] = merge_hour_record(hourly[hidx], rec)

    # Write CSV
    fieldnames = [
        "hour_index",
        "datetime_utc",
        "lat_deg",
        "lon_deg",
        "elev_m",
        "report_type",
        "u10_mps",
        "wd_deg",
        "Ta_K",
        "Td_K",
        "slp_hPa",
        "ceiling_m",
        "visibility_m",
        "cloud_total_oktas",
        "cloud_fraction",
        "solar_wm2",
        "is_day",
        "raw_line",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for h in range(HOURS_YEAR):
            rec = hourly.get(h, None)

            if rec is None:
                writer.writerow(
                    {
                        "hour_index": h,
                        "datetime_utc": "",
                        "lat_deg": "",
                        "lon_deg": "",
                        "elev_m": "",
                        "report_type": "",
                        "u10_mps": "",
                        "wd_deg": "",
                        "Ta_K": "",
                        "Td_K": "",
                        "slp_hPa": "",
                        "ceiling_m": "",
                        "visibility_m": "",
                        "cloud_total_oktas": "",
                        "cloud_fraction": "",
                        "solar_wm2": "",
                        "is_day": "",
                        "raw_line": "",
                    }
                )
                continue

            writer.writerow(
                {
                    "hour_index": h,
                    "datetime_utc": rec["datetime_utc"].strftime("%Y-%m-%d %H:%M:%S"),
                    "lat_deg": rec["lat_deg"] if rec["lat_deg"] is not None else "",
                    "lon_deg": rec["lon_deg"] if rec["lon_deg"] is not None else "",
                    "elev_m": rec["elev_m"] if rec["elev_m"] is not None else "",
                    "report_type": rec["report_type"],
                    "u10_mps": rec["u10_mps"] if rec["u10_mps"] is not None else "",
                    "wd_deg": rec["wd_deg"] if rec["wd_deg"] is not None else "",
                    "Ta_K": rec["Ta_K"] if rec["Ta_K"] is not None else "",
                    "Td_K": rec["Td_K"] if rec["Td_K"] is not None else "",
                    "slp_hPa": rec["slp_hPa"] if rec["slp_hPa"] is not None else "",
                    "ceiling_m": (
                        rec["ceiling_m"] if rec["ceiling_m"] is not None else ""
                    ),
                    "visibility_m": (
                        rec["visibility_m"] if rec["visibility_m"] is not None else ""
                    ),
                    "cloud_total_oktas": (
                        rec["cloud_total_oktas"]
                        if rec["cloud_total_oktas"] is not None
                        else ""
                    ),
                    "cloud_fraction": (
                        rec["cloud_fraction"]
                        if rec["cloud_fraction"] is not None
                        else ""
                    ),
                    "solar_wm2": (
                        rec["solar_wm2"] if rec["solar_wm2"] is not None else ""
                    ),
                    "is_day": rec["is_day"] if rec["is_day"] is not None else "",
                    "raw_line": rec["raw_line"],
                }
            )

    print(f"OK: wrote {OUTPUT_CSV}")
    print(f"Raw lines read: {total_lines}")
    print(f"Records parsed: {parsed_records}")
    print(f"Hours parsed: {len(hourly)} / {HOURS_YEAR}")
    print(f"Valid fixed ISD SLP fields: {fixed_slp_valid_count}")
    print(f"Valid METAR QNH fields: {metar_qnh_valid_count}")


if __name__ == "__main__":
    main()
