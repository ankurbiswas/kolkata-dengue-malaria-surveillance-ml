"""
02_clean_data.py

Cleans the harmonized merged_raw.csv produced by 01_merge_raw_data.py.
Handles the messiness typical of 12 years of manually-entered KMC
line-listing data: date ranges/typos, age written as "4Y6M"/"38Y"/"8 MONTHS",
sex values contaminated by shifted columns, ~2,000 free-text treatment-place
variants, and Roman-numeral borough typos.

Produces:
    data/processed/cleaned_dataset.csv   - one row per case, cleaned fields
    data/processed/cleaning_report.txt   - what was fixed / dropped and why

Run: python3 02_clean_data.py
"""
import os
import re
import difflib
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(HERE, "..", "data")   # not included in the repository (see DATA.md)
IN_PATH = os.path.join(DATA_DIR, "merged_raw.csv")
OUT_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
REPORT_PATH = os.path.join(DATA_DIR, "cleaning_report.txt")

VALID_YEAR_MIN, VALID_YEAR_MAX = 2013, 2026
VALID_AGE_MIN, VALID_AGE_MAX = 0, 110

ROMAN_BOROUGHS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII",
                  "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

MONTH_NAMES = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
               "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]

INSTITUTION_MARKERS = ["NURSING", "HOSPITAL", "CLINIC", "CENTRE", "CENTER", "COLLEGE",
                        "SANJEEVANI", "KENDRA", "SADAN", "SEVA", "MEDICAL", "HEALTH",
                        "PGIPS", "ESI", "RTIICS", "RKMSP", "CMRI", "AMRI", "ICH", "IDBG",
                        "ID & BG", "R K M", "SUPERSPECIALITY", "BED NO"]

log_lines = []


def log(msg):
    log_lines.append(str(msg))


# ---------------------------------------------------------------- Date -----
# Some cells were stored as real Excel date types (openpyxl -> python datetime
# -> "YYYY-MM-DD HH:MM:SS" after the CSV round-trip); others as free text
# "DD.MM.YYYY" (occasionally a range like "DD.MM.YYYY TO DD.MM.YYYY"). ISO
# form must be checked FIRST or its dashes get misread as a DD-MM-YY pattern.
ISO_PATTERN = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
DATE_PATTERN = re.compile(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})")


def clean_date(raw):
    if pd.isna(raw):
        return pd.NaT, "missing"
    s = str(raw).strip()

    m = ISO_PATTERN.match(s)
    if m:
        y, mo, d = m.groups()
        y = int(y)
        try:
            dt = pd.Timestamp(year=y, month=int(mo), day=int(d))
        except Exception:
            return pd.NaT, "invalid_calendar_date"
        if not (VALID_YEAR_MIN <= dt.year <= VALID_YEAR_MAX):
            return pd.NaT, "implausible_year"
        return dt, "ok"

    m = DATE_PATTERN.search(s)  # takes the FIRST date found (handles "X TO Y" ranges)
    if not m:
        return pd.NaT, "unparseable"
    d, mo, y = m.groups()
    y = int(y)
    if y < 100:
        y += 2000
    try:
        dt = pd.Timestamp(year=y, month=int(mo), day=int(d))
    except Exception:
        return pd.NaT, "invalid_calendar_date"
    if not (VALID_YEAR_MIN <= dt.year <= VALID_YEAR_MAX):
        return pd.NaT, "implausible_year"
    return dt, "ok"


# --------------------------------------------------------------- Month -----
def normalize_month_text(raw):
    if pd.isna(raw):
        return None
    s = re.sub(r"[^A-Za-z]", "", str(raw)).upper()
    if not s:
        return None
    match = difflib.get_close_matches(s, MONTH_NAMES, n=1, cutoff=0.6)
    return match[0] if match else None


# ----------------------------------------------------------------- Age -----
AGE_YM_PATTERN = re.compile(r"^(\d{1,3})\s*Y\s*(\d{1,2})?\s*M?$")
AGE_M_PATTERN = re.compile(r"^(\d{1,3})\s*(?:M|MONTHS?)$")


def clean_age(raw):
    if pd.isna(raw):
        return np.nan, "missing"
    s = str(raw).strip().upper()
    s_compact = re.sub(r"\s+", "", s)

    # plain numeric
    try:
        val = float(s)
        if VALID_AGE_MIN <= val <= VALID_AGE_MAX:
            return val, "ok"
        else:
            return np.nan, "outlier_out_of_range"
    except ValueError:
        pass

    m = AGE_YM_PATTERN.match(s_compact)
    if m:
        years = int(m.group(1))
        months = int(m.group(2)) if m.group(2) else 0
        val = years + months / 12.0
        if VALID_AGE_MIN <= val <= VALID_AGE_MAX:
            return round(val, 3), "converted_years_months"
        return np.nan, "outlier_out_of_range"

    m = AGE_M_PATTERN.match(s_compact)
    if m:
        months = int(m.group(1))
        val = months / 12.0
        return round(val, 3), "converted_months_only"

    if s_compact in ("M", "F", "MALE", "FEMALE"):
        return np.nan, "shifted_column_sex_value"

    return np.nan, "unparseable_text"


# ----------------------------------------------------------------- Sex -----
def clean_sex(raw):
    if pd.isna(raw):
        return None, "missing"
    s = re.sub(r"[^A-Za-z]", "", str(raw)).upper()
    if s in ("M", "MALE"):
        return "M", "ok"
    if s in ("F", "FEMALE"):
        return "F", "ok"
    return None, "unrecognized"


# -------------------------------------------------------------- Borough -----
def clean_borough(raw):
    if pd.isna(raw):
        return None, "missing"
    s = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    if s in ROMAN_BOROUGHS:
        return s, "ok"
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 16:
            return ROMAN_BOROUGHS[n - 1], "converted_arabic_numeral"
    return None, "unrecognized"


# -------------------------------------------------------------- Ward -----
def clean_ward(raw):
    if pd.isna(raw):
        return np.nan, "missing"
    s = re.sub(r"[^0-9]", "", str(raw))
    if not s:
        return np.nan, "unrecognized"
    val = int(s)
    if 1 <= val <= 144:
        return val, "ok"
    return np.nan, "out_of_range"


# --------------------------------------------------------- Treatment -----
def normalize_treatment(v):
    if pd.isna(v):
        return None
    v = str(v).upper().strip()
    v = re.sub(r"[^A-Z0-9 ]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def classify_treatment(raw):
    n = normalize_treatment(raw)
    if not n:
        return None, "missing"
    if any(marker in n for marker in INSTITUTION_MARKERS):
        return "Hospital", "ok"
    if n.startswith("HOSPITAL"):
        return "Hospital", "ok"
    ratio = difflib.SequenceMatcher(None, n, "HOME TREATMENT").ratio()
    if ratio > 0.55 or n in ("HOME", "HOME TREAT", "HOMETREATMENT"):
        return "Home", "ok"
    if n.startswith("HOME") and len(n) <= 20:
        return "Home", "ok"
    return "Hospital", "ok"  # default: treated as an institution/place name


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    n0 = len(df)
    log(f"Loaded merged_raw.csv: {n0} rows\n")

    # --- Date ---
    parsed = df["Date_Raw"].apply(clean_date)
    df["Date"] = [p[0] for p in parsed]
    date_flag = [p[1] for p in parsed]
    log("Date cleaning outcomes:")
    log(pd.Series(date_flag).value_counts().to_string())
    log("")

    # --- Month (prefer parsed Date; fallback to text) ---
    month_from_date = df["Date"].dt.month_name().str.upper()
    month_from_text = df["Month_Raw"].apply(normalize_month_text)
    df["Month"] = month_from_date.fillna(month_from_text)
    month_source = np.where(month_from_date.notna(), "from_date",
                             np.where(month_from_text.notna(), "from_text", "missing"))
    log("Month source breakdown:")
    log(pd.Series(month_source).value_counts().to_string())
    log("")

    # --- Age ---
    parsed = df["Age_Raw"].apply(clean_age)
    df["Age"] = [p[0] for p in parsed]
    age_flag = [p[1] for p in parsed]
    log("Age cleaning outcomes:")
    log(pd.Series(age_flag).value_counts().to_string())
    log("")

    # --- Sex ---
    parsed = df["Sex_Raw"].apply(clean_sex)
    df["Sex"] = [p[0] for p in parsed]
    sex_flag = [p[1] for p in parsed]
    log("Sex cleaning outcomes:")
    log(pd.Series(sex_flag).value_counts().to_string())
    log("")

    # --- Ward ---
    parsed = df["Ward_Raw"].apply(clean_ward)
    df["Ward"] = [p[0] for p in parsed]
    ward_flag = [p[1] for p in parsed]
    log("Ward cleaning outcomes:")
    log(pd.Series(ward_flag).value_counts().to_string())
    log("")

    # --- Borough ---
    parsed = df["Borough_Raw"].apply(clean_borough)
    df["Borough"] = [p[0] for p in parsed]
    borough_flag = [p[1] for p in parsed]
    log("Borough cleaning outcomes:")
    log(pd.Series(borough_flag).value_counts().to_string())
    log("")

    # --- Treatment type (Home vs Hospital) ---
    parsed = df["Treatment_Raw"].apply(classify_treatment)
    df["Treatment_Type"] = [p[0] for p in parsed]
    treatment_flag = [p[1] for p in parsed]
    log("Treatment_Type derivation outcomes:")
    log(pd.Series(treatment_flag).value_counts().to_string())
    log("Treatment_Type value counts:")
    log(df["Treatment_Type"].value_counts(dropna=False).to_string())
    log("")

    keep_cols = ["Date", "Month", "Age", "Sex", "Ward", "Borough", "Treatment_Type",
                 "Disease", "Year", "Source_Sheet"]
    clean_df = df[keep_cols].copy()
    clean_df.to_csv(OUT_PATH, index=False)

    log(f"\nFinal cleaned dataset: {len(clean_df)} rows -> {OUT_PATH}")
    log("\nMissingness in final cleaned columns:")
    log(clean_df.isna().mean().round(4).to_string())

    log("\nYear x Disease row counts (should match merge step exactly):")
    log(clean_df.pivot_table(index="Year", columns="Disease", values="Source_Sheet", aggfunc="size", fill_value=0).to_string())

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
