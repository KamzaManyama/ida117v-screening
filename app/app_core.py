"""Core logic for the IDA117V food-insecurity screening interface (Q7).
Loads the saved, locked pipeline (preprocessing + tuned MLP + threshold) and
handles validation, prediction and plain-language explanation."""
import json, os, sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ida117v_pipeline import ASSET_COLS, RAW_FEATURES  # noqa: E402

BUNDLE = joblib.load(os.path.join(ROOT, "outputs", "models", "final_pipeline.joblib"))
UI = json.load(open(os.path.join(ROOT, "outputs", "models", "ui_reference.json")))
PIPE, THR = BUNDLE["pipeline"], BUNDLE["threshold"]
REF, ALLOWED = UI["reference_household"], UI["allowed_values"]

ASSET_LABELS = {c: c.replace("hhasset_", "").replace("_", " ").replace("sattv", "satellite TV")
                .replace("prepaidmeter", "prepaid electricity meter").replace("loadshedding", "load-shedding backup")
                .replace("rental short", "short-term rental property").replace("rental long", "long-term rental property")
                .replace("property", "other property").replace("cell", "cellphone").replace("tv", "TV").replace("dvd", "DVD player")
                for c in ASSET_COLS}
INCOME_SOURCES = {"income_wages": "Wages / salary", "income_grant": "Social grants", "income_pension": "Pension",
                  "income_business": "Own business", "income_remittance": "Money from family elsewhere", "income_none": "No income source"}
GRANTS = {"grants_csg": "Child support grants", "grants_oag": "Old-age grants", "grants_disability": "Disability grants"}
SHOCKS = {"negevent_incomeloss": "Lost income / job", "negevent_hospital": "Hospitalisation", "negevent_crime_violent": "Violent crime",
          "negevent_fire": "Fire", "negevent_flooding": "Flooding"}
SPENDING = {"exp_energy": "Energy (electricity, gas, paraffin)", "exp_transport": "Transport", "exp_acc": "Rent / bond"}
SELECTS = {"suburb": "Suburb", "dwellingtype": "Dwelling type", "material_walls": "Main wall material", "tenure": "Tenure",
           "hhincome": "Monthly household income", "electricity": "Electricity connection", "water_piped": "Piped water access",
           "toilet": "Toilet facility"}
INCOME_ORDER = ["No income", "R1 - R400", "R401 - R800", "R801 - R1 600", "R1 601 - R3 200", "R3 201 - R6 400",
                "R6 401 - R12 800", "R12 801 - R25 600", "R25 601 - R51 200", "R51 201 - R102 400",
                "R102 401 - R204 800", "R204 801 or more", "Don't know"]

# groups of user-entered fields used for the explanation (what-if: reset group to a typical household)
EXPLAIN_GROUPS = {
    "dwelling type and wall material": ["dwellingtype", "material_walls", "tenure"],
    "household income band": ["hhincome"],
    "income sources": list(INCOME_SOURCES),
    "grants received": list(GRANTS),
    "household assets owned": ASSET_COLS,
    "basic services (water, toilet, electricity)": ["water_piped", "toilet", "electricity"],
    "recent negative events": list(SHOCKS),
    "spending on energy, transport and rent": list(SPENDING),
    "household size and children": ["hhsize", "hhchildren"],
    "suburb": ["suburb"],
}


def options(field):
    vals = ALLOWED.get(field, [])
    if field == "hhincome":
        return [v for v in INCOME_ORDER if v in vals]
    return vals


def validate(form):
    """Return (record, errors). `form` maps field -> string (checkboxes: present = 'Yes')."""
    errors, rec = [], dict(REF)
    for f, label in SELECTS.items():
        v = (form.get(f) or "").strip()
        if not v:
            errors.append(f"{label}: please choose a value.")
        elif v not in ALLOWED.get(f, []):
            errors.append(f"{label}: '{v}' is not a recognised option.")
        else:
            rec[f] = v

    def integer(f, label, lo, hi):
        v = (form.get(f) or "").strip()
        try:
            n = int(v)
        except ValueError:
            errors.append(f"{label}: enter a whole number (you entered '{v}')."); return None
        if not lo <= n <= hi:
            errors.append(f"{label}: must be between {lo} and {hi} (you entered {n})."); return None
        rec[f] = n; return n

    size = integer("hhsize", "Household size", 1, 20)
    kids = integer("hhchildren", "Number of children", 0, 19)
    if size is not None and kids is not None and kids >= size:
        errors.append("Number of children must be smaller than household size (the tool needs at least one adult).")
    for f, label in GRANTS.items():
        n = integer(f, label, 0, 10)
        if n is not None and size is not None and n > size:
            errors.append(f"{label}: cannot be more than the number of people in the household.")
    for f, label in SPENDING.items():
        integer(f, f"Monthly spending on {label.lower()} (rand)", 0, 200000)
    for f in ASSET_COLS + list(SHOCKS):
        rec[f] = "Yes" if form.get(f) else "No"
    for f in INCOME_SOURCES:
        rec[f] = "Yes" if form.get(f) else "No"
    if rec["income_none"] == "Yes" and any(rec[f] == "Yes" for f in INCOME_SOURCES if f != "income_none"):
        errors.append("'No income source' cannot be ticked together with another income source.")
    if rec["income_none"] == "No" and not any(rec[f] == "Yes" for f in INCOME_SOURCES):
        errors.append("Tick at least one income source, or tick 'No income source'.")
    if rec.get("hhincome") == "No income" and rec["income_wages"] == "Yes":
        errors.append("Income band 'No income' conflicts with receiving wages.")
    rec["negevent_none"] = "No" if any(rec[f] == "Yes" for f in SHOCKS) else "Yes"
    return rec, errors


def _proba(records):
    return PIPE.predict_proba(pd.DataFrame(records)[RAW_FEATURES])[:, 1]


def predict(rec):
    p = float(_proba([rec])[0])
    # what-if explanation: how much does the score change if this group were typical?
    trials = []
    for name, cols in EXPLAIN_GROUPS.items():
        alt = dict(rec); alt.update({c: REF[c] for c in cols}); trials.append(alt)
    effects = {n: p - float(q) for n, q in zip(EXPLAIN_GROUPS, _proba(trials))}
    up = [n for n, e in sorted(effects.items(), key=lambda x: -x[1]) if e > 0.02][:3]
    down = [n for n, e in sorted(effects.items(), key=lambda x: x[1]) if e < -0.02][:3]
    flagged = p >= THR
    band = "High" if p >= 0.75 else "Elevated" if flagged else "Moderate" if p >= 0.30 else "Low"
    text = (f"The model gives a score of {p:.0%}. " +
            ("This is above the screening threshold, so the household profile is flagged for food-security outreach. "
             if flagged else "This is below the screening threshold, so the household profile is not flagged. "))
    if up: text += "Compared with a typical surveyed household, the score is pushed UP mainly by: " + "; ".join(up) + ". "
    if down: text += "It is pushed DOWN mainly by: " + "; ".join(down) + ". "
    return {"probability": p, "flagged": flagged, "band": band, "threshold": THR, "explanation": text,
            "effects": {k: round(v, 3) for k, v in effects.items()}}


def predict_csv(df):
    """Batch scoring for an uploaded CSV with raw survey columns."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if len(missing) == len(RAW_FEATURES):
        raise ValueError("None of the expected survey columns were found. Use the column names from the data dictionary.")
    if len(df) > 20000:
        raise ValueError("Please upload at most 20,000 rows at a time.")
    filled = df.copy()
    for c in missing:
        filled[c] = REF[c]
    p = _proba(filled.to_dict("records"))
    out = pd.DataFrame({"row": range(1, len(df) + 1), "score": p.round(3), "flagged": p >= THR})
    return out, missing
