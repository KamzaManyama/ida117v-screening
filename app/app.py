"""IDA117V — Household food-insecurity screening tool (Q7 user interface, Flask).
Run from the project root:   python app/app.py      then open http://127.0.0.1:5000
Pages: Screen a household (/, quick or full form), Suburb lookup (/suburbs), Batch scoring (/batch),
Model & evidence (/model), Guide & deploy (/guide). All predictions use the saved pipeline loaded in app_core.py.
"""
import io
import json
import os

import pandas as pd
from flask import Flask, Response, render_template_string, request

from app_core import (ASSET_COLS, ASSET_LABELS, GRANTS, INCOME_SOURCES, RAW_FEATURES, REF, SELECTS, SHOCKS, SPENDING, THR,
                      options, predict, predict_csv, validate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# ------------------------------------------------------------------ locked test-set evidence (Q5.3, Q6.2, Q8.1)
# Values from outputs/tables/ (q5_test_metrics.csv, q8_fairness_dwelling.csv, q6_mcnemar.json). Files are read when
# present; these defaults are the same numbers, so the hosted app works without the tables folder.
EVID = {"recall": 0.840, "precision": 0.309, "roc_auc": 0.848, "pr_auc": 0.422, "macro_f1": 0.637, "accuracy": 0.732,
        "test_n": 1238, "test_pos": 163, "prevalence": 0.132,
        "fair": [["Formal", 891, 0.074, 0.224, 0.788, 0.26], ["Backyard (ADI)", 187, 0.235, 0.578, 0.818, 0.333],
                 ["Informal settlement", 160, 0.331, 0.844, 0.925, 0.363]],
        "mcnemar": {"comparator": "Logistic Regression", "both_correct": 835, "champion_only_correct": 71,
                    "comparator_only_correct": 67, "both_wrong": 265, "p": 0.798}}
try:
    T = os.path.join(ROOT, "outputs", "tables")
    m = pd.read_csv(os.path.join(T, "q5_test_metrics.csv"), index_col=0).iloc[:, 0]
    for k in ["recall", "precision", "roc_auc", "pr_auc", "macro_f1", "accuracy"]:
        EVID[k] = float(m[f"{k} (threshold {THR:.2f})"])
    f = pd.read_csv(os.path.join(T, "q8_fairness_dwelling.csv"))
    EVID["fair"] = f[["group", "households", "prevalence", "flagged_rate", "recall", "precision"]].values.tolist()
    mc = json.load(open(os.path.join(T, "q6_mcnemar.json")))
    EVID["mcnemar"] = dict(mc["table"], comparator=mc["comparator"], p=mc["p_value_exact_binomial"])
except Exception:
    pass

# ------------------------------------------------------------------ Q7.4 demonstration cases (same as app/demo_cases.py)
def _pick(f, s):
    return next((o for o in options(f) if s.lower() in o.lower()), options(f)[0] if options(f) else "")

_base = {"hhsize": "4", "hhchildren": "2", "grants_csg": "0", "grants_oag": "0", "grants_disability": "0",
         "exp_energy": "800", "exp_transport": "1000", "exp_acc": "0"}
DEMOS = [
    ("Formal house, middle income", "valid", dict(_base, suburb=_pick("suburb", "Bothasig"), dwellingtype="Formal",
        material_walls=_pick("material_walls", "Bricks"), tenure=_pick("tenure", "fully paid"), hhincome=_pick("hhincome", "R25 601"),
        water_piped=_pick("water_piped", "inside the dwelling"), toilet=_pick("toilet", "Flush toilet connected to a public"),
        electricity=_pick("electricity", "In-house prepaid"), income_wages="on", hhasset_fridge="on", hhasset_stove="on",
        hhasset_washer="on", hhasset_computer="on", hhasset_vehicle="on", hhasset_tv="on", hhasset_cell="on",
        hhasset_geyser="on", hhasset_vacuum="on", exp_transport="2500")),
    ("Informal settlement, grants only", "valid", dict(_base, hhsize="5", hhchildren="3", suburb=_pick("suburb", "Delft"),
        dwellingtype="Informal settlement", material_walls=_pick("material_walls", "Corrugated"), tenure=_pick("tenure", "rent-free"),
        hhincome=_pick("hhincome", "R801 - R1 600"), water_piped=_pick("water_piped", "community"), toilet=_pick("toilet", "chemical"),
        electricity=_pick("electricity", "neighbour"), income_grant="on", grants_csg="3", hhasset_cell="on",
        negevent_incomeloss="on", exp_energy="300", exp_transport="400")),
    ("Backyard dwelling, wages", "valid", dict(_base, hhsize="3", hhchildren="1", suburb=_pick("suburb", "Hanover Park"),
        dwellingtype="ADI", material_walls=_pick("material_walls", "Wood"), tenure=_pick("tenure", "private indiv"),
        hhincome=_pick("hhincome", "R3 201"), water_piped=_pick("water_piped", "inside the yard"),
        toilet=_pick("toilet", "Flush toilet connected to a public"), electricity=_pick("electricity", "In-house prepaid"),
        income_wages="on", grants_csg="1", income_grant="on", hhasset_fridge="on", hhasset_stove="on", hhasset_tv="on",
        hhasset_cell="on", exp_acc="1500")),
    ("Invalid input (shows errors)", "invalid", dict(_base, hhsize="2", hhchildren="2", suburb="", dwellingtype="Formal",
        material_walls=_pick("material_walls", "Bricks"), tenure=_pick("tenure", "fully paid"), hhincome="No income",
        water_piped=_pick("water_piped", "inside the dwelling"), toilet=_pick("toilet", "Flush toilet connected to a public"),
        electricity=_pick("electricity", "In-house prepaid"), income_wages="on", income_none="on", exp_energy="-50")),
]

GUIDE = {  # Q7.3 feature guidance shown under each field
    "suburb": "Main place or suburb of the dwelling (163 surveyed areas).",
    "dwellingtype": "Formal house/flat, ADI = backyard dwelling, or informal settlement.",
    "material_walls": "What most of the outside walls are made of.",
    "tenure": "Who owns or rents the dwelling.",
    "hhincome": "Total monthly income of everyone in the household, before deductions.",
    "water_piped": "Where the household gets piped water.",
    "toilet": "The toilet the household mainly uses.",
    "electricity": "How the dwelling gets electricity.",
}

# ------------------------------------------------------------------ suburb lookup + quick-form defaults
# ui_lookup.json holds survey aggregates only (no household records): suburb rates and model-flagged shares for
# suburbs with at least 20 surveyed households, and the most common answers per dwelling type.
LOOKUP = json.load(open(os.path.join(ROOT, "outputs", "models", "ui_lookup.json")))
SUBURBS = {s["name"]: s for s in LOOKUP["suburbs"]}
DEFAULTS = LOOKUP["dwelling_defaults"]
SEARCH_FIELDS = ["suburb", "material_walls", "tenure", "water_piped", "toilet", "electricity"]   # long lists: type to search
QUICK_ASSETS = ["hhasset_fridge", "hhasset_stove", "hhasset_washer", "hhasset_tv", "hhasset_vehicle", "hhasset_computer"]
QUICK_INCOME = ["income_wages", "income_grant", "income_none"]


def canon(field, value):
    """Match typed text to a known answer, ignoring case and extra spaces (so 'delft' finds 'Delft')."""
    v = " ".join((value or "").split()).lower()
    for o in options(field):
        if " ".join(o.split()).lower() == v:
            return o
    return (value or "").strip()


def expand_quick(form):
    """Turn the 9-question quick form into a full record: unasked answers = most common for that dwelling type."""
    dt = form.get("dwellingtype") if form.get("dwellingtype") in DEFAULTS else "Formal"
    d = DEFAULTS[dt]
    full = {f: d[f] for f in ["material_walls", "tenure", "water_piped", "toilet", "electricity"]}
    full.update({f: str(d[f]) for f in ["exp_energy", "exp_transport", "exp_acc"]})
    full.update({"grants_oag": "0", "grants_disability": "0"})
    for f in ASSET_COLS:
        if f not in QUICK_ASSETS and d.get(f) == "Yes":
            full[f] = "on"
    for f in ["suburb", "dwellingtype", "hhsize", "hhchildren", "hhincome", "grants_csg"] + QUICK_ASSETS + QUICK_INCOME + ["negevent_incomeloss"]:
        if form.get(f):
            full[f] = form[f]
    assumed = [(SELECTS[f], full[f]) for f in ["material_walls", "tenure", "water_piped", "toilet", "electricity"]]
    assumed.append(("Other assets owned", ", ".join(ASSET_LABELS[f] for f in ASSET_COLS if f not in QUICK_ASSETS and full.get(f)) or "none"))
    assumed.append(("Monthly energy / transport / rent", f"R{full['exp_energy']} / R{full['exp_transport']} / R{full['exp_acc']}"))
    return full, assumed, dt


ICONS = {
    "suburbs": '<path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "screen": '<path d="M4 5h16v14H4z"/><path d="M8 9h8M8 13h5"/>',
    "batch": '<path d="M6 3h9l3 3v15H6z"/><path d="M9 11h6M9 15h6M9 7h3"/>',
    "model": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "guide": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "test": '<path d="M5 4h14v16H5z"/><path d="M8 8h8M8 12h8M8 16h4"/>',
    "people": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><path d="M16 4.5a3 3 0 0 1 0 6M18 14c2 .7 3 2.8 3 6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "curve": '<path d="M3 21V3M3 21h18"/><path d="M5 18C8 8 13 6 20 5"/>',
    "gauge": '<path d="M4 16a8 8 0 1 1 16 0"/><path d="M12 16l4-5"/>',
}

PAGE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{{ title }} · Food Security Screening</title>
<style>
/* Layout: white side nav (left) + light-grey canvas; KPI cards on top, working cards in a grid. Black and white only. */
:root{--bg:#f2f2f2;--card:#fff;--ink:#000;--ink-2:#333;--muted:#6b6b6b;--line:#dcdcdc;--tile:#f5f5f5;
  --f:Arial,Helvetica,sans-serif;--fs-h1:30px;--fs-h2:20px;--fs-h3:16px;--fs-p:15px;--fs-s:13px;--fs-l:11.5px;--side:232px}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--f);font-size:var(--fs-p);line-height:1.5}
h1,h2,h3{margin:0;font-weight:700;line-height:1.2}
h1{font-size:var(--fs-h1)} h2{font-size:var(--fs-h2)} h3{font-size:var(--fs-h3)}
p{margin:0}
a{color:inherit}
:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.app{display:grid;grid-template-columns:var(--side) minmax(0,1fr);min-height:100vh;background:linear-gradient(90deg,var(--card) var(--side),var(--bg) var(--side))}
/* side nav */
.side{background:var(--card);border-right:1px solid var(--line);position:sticky;top:0;height:100vh;display:flex;flex-direction:column;
  padding:96px 0 20px;padding-left:env(safe-area-inset-left,0px)}
.logo{font-size:30px;font-weight:700;letter-spacing:-.02em;padding:0 28px 6px;display:flex;align-items:baseline;gap:2px}
.logo i{width:9px;height:9px;background:var(--ink);display:inline-block;border-radius:50%}
.logo-sub{font-size:var(--fs-l);color:var(--muted);padding:0 28px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:44px}
.nav{display:flex;flex-direction:column}
.nav a{display:flex;align-items:center;gap:14px;padding:13px 28px;text-decoration:none;font-size:15px;color:var(--ink-2);border-left:3px solid transparent}
.nav a svg{width:19px;height:19px;flex:none}
.nav a:hover{background:var(--tile)}
.nav a[aria-current=page]{border-left-color:var(--ink);font-weight:700;color:var(--ink);background:var(--tile)}
.side-foot{margin-top:auto;padding:0 28px;font-size:var(--fs-l);color:var(--muted);line-height:1.5}
.side-foot b{color:var(--ink)}
svg.i{fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
/* main */
.main{min-width:0;padding:30px 34px 44px;padding-right:max(34px,env(safe-area-inset-right,0px));display:grid;gap:22px;align-content:start}
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
.head p{color:var(--muted);font-size:var(--fs-s);margin-top:6px}
.pill{display:inline-flex;align-items:center;gap:8px;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:7px 14px;font-size:var(--fs-s);white-space:nowrap}
.pill b{font-weight:700}
.pill .dot{width:8px;height:8px;border-radius:50%;background:var(--ink)}
/* KPI cards */
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
.kpi{background:var(--card);padding:18px 18px 0;display:grid;grid-template-columns:1fr auto;gap:4px 10px;position:relative;box-shadow:0 1px 2px #0000000d}
.kpi .lab{font-size:var(--fs-l);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);grid-column:1}
.kpi .val{font-size:28px;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums;grid-column:1}
.kpi .sub{font-size:var(--fs-l);color:var(--muted);grid-column:1;padding-bottom:16px}
.kpi svg{grid-column:2;grid-row:1/4;width:30px;height:30px;color:#b5b5b5;align-self:center}
.kpi::after{content:"";position:absolute;left:18px;bottom:0;height:3px;width:calc((100% - 36px) * var(--f,.4));background:var(--ink)}
/* cards */
.grid{display:grid;gap:22px;grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr);align-items:start}
.grid.even{grid-template-columns:repeat(2,minmax(0,1fr))}
.card{background:var(--card);box-shadow:0 1px 2px #0000000d;min-width:0}
.card-h{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:18px 20px 12px;border-bottom:1px solid var(--line)}
.card-h span{font-size:var(--fs-s);color:var(--muted)}
.card-b{padding:18px 20px 20px;display:grid;gap:14px}
.sticky{position:sticky;top:20px}
.sec{display:grid;gap:12px;padding-bottom:18px;border-bottom:1px solid var(--line)}
.sec:last-of-type{border-bottom:0}
.sec-t{display:flex;align-items:center;gap:10px;font-weight:700;font-size:var(--fs-h3)}
.sec-t b{width:24px;height:24px;border-radius:50%;background:var(--ink);color:#fff;font-size:12px;display:grid;place-items:center;flex:none}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,220px),1fr));gap:12px 16px}
label.f{display:grid;gap:5px;font-size:var(--fs-s);font-weight:700;min-width:0}
label.f small{font-weight:400;color:var(--muted);font-size:var(--fs-l);line-height:1.35}
select,input[type=number],input[type=file]{font:inherit;font-size:14px;padding:8px 10px;border:1px solid #bdbdbd;border-radius:4px;background:#fff;color:var(--ink);width:100%;min-width:0;font-weight:400}
select:focus,input:focus{border-color:var(--ink);outline:none;box-shadow:0 0 0 1px var(--ink)}
.checks{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,190px),1fr));gap:6px 16px}
.checks label{display:flex;gap:8px;align-items:center;font-size:14px}
.checks input{accent-color:#000;width:16px;height:16px}
.btn{font:inherit;font-size:15px;font-weight:700;background:var(--ink);color:#fff;border:1px solid var(--ink);border-radius:4px;padding:11px 20px;cursor:pointer;text-decoration:none;display:inline-flex;gap:8px;align-items:center}
.btn.ghost{background:#fff;color:var(--ink)}
.btn:hover{opacity:.85}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:var(--fs-s);border:1px solid var(--ink);border-radius:999px;padding:6px 12px;text-decoration:none;background:#fff}
.chip.dash{border-style:dashed}
.chip[aria-current=true]{background:var(--ink);color:#fff}
/* result */
.empty{color:var(--muted);font-size:14px;border:1px dashed #bdbdbd;padding:22px;text-align:center}
.verdict{font-size:22px;font-weight:700;line-height:1.25}
.verdict.flag{text-decoration:underline;text-decoration-thickness:3px;text-underline-offset:5px}
.score{display:flex;align-items:baseline;gap:10px}
.score b{font-size:56px;letter-spacing:-.02em;line-height:1;font-variant-numeric:tabular-nums}
.score span{color:var(--muted);font-size:var(--fs-s)}
.meter{position:relative;height:14px;border:1px solid var(--ink);border-radius:999px;overflow:hidden;background:var(--tile)}
.meter span{position:absolute;inset:0 auto 0 0;background:var(--ink)}
.meter-wrap{position:relative;padding-top:18px}
.meter-wrap i{position:absolute;top:0;bottom:-4px;border-left:2px dashed var(--ink)}
.meter-wrap em{position:absolute;top:0;font-style:normal;font-size:var(--fs-l);transform:translateX(-50%);white-space:nowrap;background:#fff;padding:0 3px}
.band{display:inline-block;font-size:var(--fs-l);text-transform:uppercase;letter-spacing:.08em;border:1px solid var(--ink);padding:3px 8px;border-radius:3px;font-weight:700}
.why{display:grid;gap:7px}
.why-row{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr) 46px;gap:8px;align-items:center;font-size:var(--fs-s)}
.why-row .neg,.why-row .pos{height:12px;position:relative}
.why-row .neg span{position:absolute;right:0;top:0;bottom:0;background:repeating-linear-gradient(135deg,#000 0 1.5px,#fff 1.5px 4.5px);border:1px solid #000}
.why-row .pos span{position:absolute;left:0;top:0;bottom:0;background:#000}
.why-row .pos{border-left:1px solid var(--ink)}
.why-row output{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
.why-key{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr) 46px;gap:8px;font-size:var(--fs-l);color:var(--muted)}
.why-key span:nth-child(2){text-align:right}
.note{font-size:var(--fs-s);color:var(--ink-2);line-height:1.5}
.notice{border:1px solid var(--ink);border-left:6px solid var(--ink);background:#fff;padding:14px 16px;font-size:14px;line-height:1.5}
.notice b{font-weight:700}
.err{border:2px solid var(--ink);background:repeating-linear-gradient(135deg,#f3f3f3 0 8px,#fff 8px 16px);padding:14px 18px}
.err h3{margin-bottom:6px}
.err ul{margin:0;padding-left:20px;display:grid;gap:4px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:var(--fs-s);font-variant-numeric:tabular-nums}
th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:var(--fs-l);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:700}
td:first-child,th:first-child{text-align:left}
td.l,th.l{text-align:left}
.tbl{overflow-x:auto}
.bars{display:grid;gap:10px}
.bar-row{display:grid;grid-template-columns:150px minmax(0,1fr) 56px;gap:10px;align-items:center;font-size:var(--fs-s)}
.bar-row .track{height:14px;background:var(--tile);position:relative}
.bar-row .track span{position:absolute;inset:0 auto 0 0;background:var(--ink)}
.bar-row .track span.h{background:repeating-linear-gradient(135deg,#000 0 1.5px,#fff 1.5px 4.5px);border:1px solid #000}
.bar-row b{text-align:right;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:var(--fs-s);color:var(--ink-2)}
.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-1px;background:#000}
.legend i.h{background:repeating-linear-gradient(135deg,#000 0 1.5px,#fff 1.5px 4.5px);border:1px solid #000}
.mc{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;font-size:var(--fs-s)}
.mc div{padding:12px;background:var(--tile);text-align:center}
.mc .h{background:none;color:var(--muted);padding:4px}
.mc b{display:block;font-size:24px}
pre{background:#000;color:#fff;padding:14px 16px;font-size:13px;overflow-x:auto;margin:0;line-height:1.6}
ul.list{margin:0;padding-left:20px;display:grid;gap:7px;font-size:14px}
.steps{display:grid;gap:12px;counter-reset:s;margin:0;padding:0;list-style:none}
.steps li{counter-increment:s;display:grid;grid-template-columns:28px 1fr;gap:12px;font-size:14px}
.steps li::before{content:counter(s);width:26px;height:26px;border-radius:50%;border:1px solid var(--ink);display:grid;place-items:center;font-weight:700;font-size:12px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kv div{background:var(--tile);padding:12px 14px}
.kv span{display:block;font-size:var(--fs-l);color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.kv b{font-size:22px}
.tabs-mode{display:flex;gap:8px;flex-wrap:wrap}
.tabs-mode a{font-size:14px;padding:8px 14px;border:1px solid var(--ink);border-radius:999px;text-decoration:none;background:#fff}
.tabs-mode a[aria-current=true]{background:var(--ink);color:#fff;font-weight:700}
input[type=search],input[list]{font:inherit;font-size:14px;padding:8px 10px;border:1px solid #bdbdbd;border-radius:4px;background:#fff;color:var(--ink);width:100%;min-width:0;font-weight:400}
.searchbar{display:flex;gap:10px;flex-wrap:wrap}.searchbar input{flex:1 1 260px}
.assumed{display:grid;gap:4px;font-size:var(--fs-s);color:var(--ink-2)}.assumed div{display:grid;grid-template-columns:150px 1fr;gap:8px}
.assumed b{color:var(--ink)}
.sub-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr));gap:16px}
.mix{display:flex;height:12px;border:1px solid var(--ink)}.mix span{display:block;height:100%}
.mix .m1{background:#000}.mix .m2{background:repeating-linear-gradient(135deg,#000 0 1.5px,#fff 1.5px 4.5px)}.mix .m3{background:#fff}
tr.small td{color:var(--muted)}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.grid,.grid.even{grid-template-columns:minmax(0,1fr)}.sticky{position:static}}
@media (max-width:800px){
  .app{grid-template-columns:minmax(0,1fr);background:var(--bg)}
  .side{position:sticky;top:0;z-index:10;height:auto;flex-direction:row;align-items:center;padding:10px max(16px,env(safe-area-inset-right,0px)) 10px max(16px,env(safe-area-inset-left,0px));border-right:0;border-bottom:1px solid var(--line);gap:8px}
  .nav{flex-direction:row;overflow-x:auto;scrollbar-width:none}
  .nav a{padding:8px 10px;border-left:0;border-bottom:3px solid transparent;white-space:nowrap;font-size:14px;gap:6px}
  .nav a[aria-current=page]{border-bottom-color:var(--ink)}
  .main{padding:20px max(16px,env(safe-area-inset-right,0px)) 32px max(16px,env(safe-area-inset-left,0px))}
  :root{--fs-h1:24px}
  .bar-row{grid-template-columns:110px minmax(0,1fr) 50px}
}
</style></head><body>
<div class="app">
<aside class="side">
  <nav class="nav" aria-label="Main">
    {% for key, href, label in nav %}<a href="{{ href }}" {% if key == page %}aria-current="page"{% endif %}><svg class="i" viewBox="0 0 24 24">{{ icons[key]|safe }}</svg>{{ label }}</a>{% endfor %}
  </nav>
</aside>
<main class="main">

<div class="head">
  <div><h1>{{ title }}</h1><p>{{ subtitle }}</p></div>
  <div class="pill"><span class="dot"></span>Model: <b>tuned MLP</b> · threshold <b>{{ '%.2f'|format(thr) }}</b></div>
</div>


{% if errors %}<div class="err" role="alert"><h3>Please fix the following:</h3><ul>{% for e in errors %}<li>{{e}}</li>{% endfor %}</ul></div>{% endif %}

{% macro searchfield(f, label, hint) %}<label class="f" for="{{f}}">{{ label }}
  <input id="{{f}}" name="{{f}}" list="list-{{f}}" autocomplete="off" placeholder="Start typing to search…" value="{{ form.get(f,'') }}">
  <small>{{ hint }}</small></label>{% endmacro %}
{% for f in search_fields %}<datalist id="list-{{f}}">{% for o in opt(f) %}<option value="{{o}}">{% endfor %}</datalist>{% endfor %}

{# ====================================================== SCREEN #}
{% if page == 'screen' %}
<section class="kpis">
  <div class="kpi" style="--f:{{ ev.recall }}"><span class="lab">At-risk households found</span><span class="val">{{ pct(ev.recall) }}</span><span class="sub">recall on the locked test set</span><svg class="i" viewBox="0 0 24 24">{{ icons.target|safe }}</svg></div>
  <div class="kpi" style="--f:{{ ev.precision }}"><span class="lab">Flags that are correct</span><span class="val">{{ pct(ev.precision) }}</span><span class="sub">precision · base rate {{ pct(ev.prevalence) }}</span><svg class="i" viewBox="0 0 24 24">{{ icons.people|safe }}</svg></div>
  <div class="kpi" style="--f:{{ ev.roc_auc }}"><span class="lab">Ranking quality</span><span class="val">{{ '%.3f'|format(ev.roc_auc) }}</span><span class="sub">ROC-AUC (1.0 = perfect)</span><svg class="i" viewBox="0 0 24 24">{{ icons.curve|safe }}</svg></div>
  <div class="kpi" style="--f:{{ thr }}"><span class="lab">Screening threshold</span><span class="val">{{ '%.0f'|format(thr*100) }}%</span><span class="sub">score at or above this is flagged</span><svg class="i" viewBox="0 0 24 24">{{ icons.gauge|safe }}</svg></div>
</section>

<div class="grid">
<form class="card" method="post" action="/" novalidate>
  <input type="hidden" name="mode" value="{{ mode }}">
  <div class="card-h"><h2>Household profile</h2><span>{{ '9 key questions · about 1 minute' if mode == 'quick' else 'All fields required · 7 short sections' }}</span></div>
  <div class="card-b">
    <nav class="tabs-mode" aria-label="Form length"><a href="/" {% if mode == 'quick' %}aria-current="true"{% endif %}>Quick screen · 9 questions</a><a href="/?mode=full" {% if mode == 'full' %}aria-current="true"{% endif %}>Full form · all answers</a></nav>
  {% if mode == 'quick' %}
    <div class="sec"><div class="sec-t"><b>1</b>Where and how they live</div><div class="fields">
      {{ searchfield('suburb', 'Suburb', 'Type part of the name, e.g. "Delft" or "Gugu".') }}
      <label class="f" for="dwellingtype">Dwelling type<select id="dwellingtype" name="dwellingtype"><option value="">Choose…</option>{% for o in opt('dwellingtype') %}<option {{ 'selected' if form.get('dwellingtype')==o }}>{{o}}</option>{% endfor %}</select><small>{{ guide['dwellingtype'] }}</small></label>
    </div></div>
    <div class="sec"><div class="sec-t"><b>2</b>Household and income</div><div class="fields">
      <label class="f" for="hhsize">Household size (people)<input id="hhsize" type="number" name="hhsize" min="1" max="20" value="{{ form.get('hhsize','') }}"><small>Everyone who usually sleeps here, 1–20.</small></label>
      <label class="f" for="hhchildren">Children under 18<input id="hhchildren" type="number" name="hhchildren" min="0" max="19" value="{{ form.get('hhchildren','') }}"><small>Must leave at least one adult.</small></label>
      <label class="f" for="hhincome">{{ selects['hhincome'] }}<select id="hhincome" name="hhincome"><option value="">Choose…</option>{% for o in opt('hhincome') %}<option {{ 'selected' if form.get('hhincome')==o }}>{{o}}</option>{% endfor %}</select><small>{{ guide['hhincome'] }}</small></label>
      <label class="f" for="grants_csg">Child support grants (number)<input id="grants_csg" type="number" name="grants_csg" min="0" max="10" value="{{ form.get('grants_csg','0') }}"><small>0 if none.</small></label>
    </div>
      <div class="checks">{% for f in quick_income %}<label><input type="checkbox" name="{{f}}" {{ 'checked' if form.get(f) }}> {{ income[f] }}</label>{% endfor %}<label><input type="checkbox" name="negevent_incomeloss" {{ 'checked' if form.get('negevent_incomeloss') }}> Lost income or a job this year</label></div></div>
    <div class="sec"><div class="sec-t"><b>3</b>Which of these do they own?</div>
      <div class="checks">{% for f in quick_assets %}<label><input type="checkbox" name="{{f}}" {{ 'checked' if form.get(f) }}> {{ asset_labels[f]|capitalize }}</label>{% endfor %}</div></div>
    <p class="note">Everything else (walls, tenure, water, toilet, electricity, other assets, spending) is set to what is most common for the chosen dwelling type. The result lists exactly what was assumed. Need more detail? Use the <a href="/?mode=full">full form</a>.</p>
    <div><button class="btn" type="submit">Assess household profile →</button></div>
  {% else %}
    <div class="sec"><div class="sec-t"><b>1</b>Location and dwelling</div><div class="fields">
      {{ searchfield('suburb', selects['suburb'], guide['suburb']) }}
      <label class="f" for="dwellingtype">{{ selects['dwellingtype'] }}<select id="dwellingtype" name="dwellingtype"><option value="">Choose…</option>{% for o in opt('dwellingtype') %}<option {{ 'selected' if form.get('dwellingtype')==o }}>{{o}}</option>{% endfor %}</select><small>{{ guide['dwellingtype'] }}</small></label>
      {{ searchfield('material_walls', selects['material_walls'], guide['material_walls']) }}
      {{ searchfield('tenure', selects['tenure'], guide['tenure']) }}</div></div>
    <div class="sec"><div class="sec-t"><b>2</b>Household</div><div class="fields">
      <label class="f" for="hhsize">Household size (people)<input id="hhsize" type="number" name="hhsize" min="1" max="20" value="{{ form.get('hhsize','') }}"><small>Everyone who usually sleeps here, 1–20.</small></label>
      <label class="f" for="hhchildren">Children under 18<input id="hhchildren" type="number" name="hhchildren" min="0" max="19" value="{{ form.get('hhchildren','') }}"><small>Must leave at least one adult.</small></label>
      <label class="f" for="hhincome">{{ selects['hhincome'] }}<select id="hhincome" name="hhincome"><option value="">Choose…</option>{% for o in opt('hhincome') %}<option {{ 'selected' if form.get('hhincome')==o }}>{{o}}</option>{% endfor %}</select><small>{{ guide['hhincome'] }}</small></label>
    </div></div>
    <div class="sec"><div class="sec-t"><b>3</b>Income sources and grants</div>
      <div class="checks">{% for f,l in income.items() %}<label><input type="checkbox" name="{{f}}" {{ 'checked' if form.get(f) }}> {{l}}</label>{% endfor %}</div>
      <div class="fields">{% for f,l in grants.items() %}<label class="f" for="{{f}}">{{l}} (number)<input id="{{f}}" type="number" name="{{f}}" min="0" max="10" value="{{ form.get(f,'0') }}"></label>{% endfor %}</div></div>
    <div class="sec"><div class="sec-t"><b>4</b>Assets owned</div>
      <div class="checks">{% for f in assets %}<label><input type="checkbox" name="{{f}}" {{ 'checked' if form.get(f) }}> {{ asset_labels[f]|capitalize }}</label>{% endfor %}</div></div>
    <div class="sec"><div class="sec-t"><b>5</b>Basic services</div><div class="fields">
      {% for f in ['water_piped','toilet','electricity'] %}{{ searchfield(f, selects[f], guide[f]) }}{% endfor %}</div></div>
    <div class="sec"><div class="sec-t"><b>6</b>Events in the past 12 months</div>
      <div class="checks">{% for f,l in shocks.items() %}<label><input type="checkbox" name="{{f}}" {{ 'checked' if form.get(f) }}> {{l}}</label>{% endfor %}</div></div>
    <div class="sec"><div class="sec-t"><b>7</b>Monthly spending (rand)</div><div class="fields">
      {% for f,l in spending.items() %}<label class="f" for="{{f}}">{{l}}<input id="{{f}}" type="number" name="{{f}}" min="0" max="200000" value="{{ form.get(f,'0') }}"><small>0–200 000</small></label>{% endfor %}</div></div>
    <p class="note">Other survey answers (amenity use, cooking fuel, internet access and so on) are set to those of a typical surveyed household.</p>
    <div><button class="btn" type="submit">Assess household profile →</button></div>
  {% endif %}
  </div>
</form>

<div class="sticky" style="display:grid;gap:22px">
  <div class="card">
    <div class="card-h"><h2>Result</h2><span>{{ 'from the saved pipeline' if result else 'waiting for input' }}</span></div>
    <div class="card-b">
    {% if result %}
      <p class="verdict {{ 'flag' if result.flagged }}">{{ 'Flagged: likely at risk of food insecurity' if result.flagged else 'Not flagged: lower likelihood of food insecurity' }} <span class="band">({{ result.band }} score)</span></p>
      <div class="score"><b>{{ '%.0f'|format(result.probability*100) }}%</b><span>Model score: {{ '%.0f'|format(result.probability*100) }}%</span></div>
      <div class="meter-wrap"><i style="left:{{ thr*100 }}%"></i><em style="left:{{ thr*100 }}%">threshold {{ '%.0f'|format(thr*100) }}%</em>
        <div class="meter"><span style="width:{{ (result.probability*100)|round(1) }}%"></span></div></div>
      <p class="note">The score ranks households for outreach. It is not a guaranteed probability.</p>
      <h3>What moved the score</h3>
      <div class="why-key"><span>answer group</span><span>pushes down</span><span>pushes up</span><span></span></div>
      <div class="why">{% for name, e in why %}<div class="why-row"><span>{{ name|capitalize }}</span>
        <div class="neg">{% if e < 0 %}<span style="width:{{ (-e/wmax*100)|round(1) }}%"></span>{% endif %}</div>
        <div class="pos">{% if e > 0 %}<span style="width:{{ (e/wmax*100)|round(1) }}%"></span>{% endif %}</div>
        <output>{{ '%+.0f'|format(e*100) }}</output></div>{% endfor %}</div>
      <p class="note">{{ result.explanation }}</p>
      <p class="note">Explanation method: each group of answers is reset to a typical surveyed household and the change in score is measured (in percentage points).</p>
      {% if assumed %}<h3>Filled in for you ({{ assumed_dt }} home)</h3><div class="assumed">{% for k, v in assumed %}<div><span>{{ k }}</span><b>{{ v }}</b></div>{% endfor %}</div>{% endif %}
      {% if sub_ctx %}<p class="note"><b>{{ sub_ctx.name }}:</b> {{ pct(sub_ctx.rate) }} of {{ sub_ctx.households }} surveyed households reported adult hunger (city {{ pct(city.rate) }}). <a href="/suburbs?q={{ sub_ctx.name|urlencode }}">Suburb details →</a></p>{% endif %}
    {% else %}
      <div class="empty">Fill in the household profile, or load one of the demo cases below, then press <b>Assess household profile</b>.</div>
    {% endif %}
    </div>
  </div>
  <div class="card">
    <div class="card-h"><h2>Demo cases</h2><span>3 valid + 1 invalid</span></div>
    <div class="card-b"><div class="chips">{% for i, d in demos %}<a class="chip {{ 'dash' if d[1]=='invalid' }}" href="/?demo={{i}}" {% if demo == i %}aria-current="true"{% endif %}>{{ i }}. {{ d[0] }}</a>{% endfor %}</div>
      <p class="note">Each case fills the form and runs it through the saved pipeline, exactly like a user would.</p></div>
  </div>
  <div class="card">
    <div class="card-h"><h2>What the classes mean</h2></div>
    <div class="card-b"><div style="display:grid;gap:8px;font-size:14px">
      <p><b>Flagged</b>: score ≥ {{ '%.0f'|format(thr*100) }}%. The profile resembles households where an adult went hungry. Prioritise for an outreach visit.</p>
      <p><b>Not flagged</b>: lower likelihood. It does <b>not</b> prove the household is food secure.</p>
      <p>Bands: Low &lt; 30% · Moderate 30–{{ '%.0f'|format(thr*100) }}% · Elevated {{ '%.0f'|format(thr*100) }}–75% · High ≥ 75%.</p></div></div>
  </div>
</div>
</div>
{% endif %}

{# ====================================================== SUBURBS #}
{% if page == 'suburbs' %}
<form class="card" method="get" action="/suburbs" role="search">
  <div class="card-h"><h2>Find a suburb</h2><span>{{ lookup.suburbs|length }} suburbs · survey 2024</span></div>
  <div class="card-b"><div class="searchbar"><input type="search" name="q" list="list-suburb" value="{{ q }}" placeholder="Type a suburb, e.g. Delft, Gugulethu, Langa" aria-label="Suburb name" autocomplete="off"><button class="btn" type="submit">Search</button>{% if q %}<a class="btn ghost" href="/suburbs">Clear</a>{% endif %}</div>
  <p class="note">Shows how many surveyed households reported an adult going hungry, and how many the model would flag. Suburbs with fewer than 20 surveyed households are not shown ({{ lookup.hidden_small }} hidden). The model flags more households than report hunger because it is set to miss as few hungry households as possible (it finds 84% of them).</p></div>
</form>
{% if q and not matches %}<div class="err" role="alert"><h3>No suburb matches "{{ q }}"</h3><ul><li>Check the spelling, or type only part of the name: "Delft" finds all four Delft areas. Large townships are split into smaller survey areas.</li></ul></div>{% endif %}
{% if matches %}
<div class="sub-cards">{% for s in matches[:6] %}
  <div class="card"><div class="card-h"><h2>{{ s.name }}</h2><span>{{ s.households }} households surveyed{% if s.households < 40 %} · small sample{% endif %}</span></div>
  <div class="card-b">
    <div class="kv"><div><span>Adult hunger (survey)</span><b>{{ pct(s.rate) }}</b></div><div><span>Flagged by model</span><b>{{ pct(s.flagged) }}</b></div></div>
    <div class="bar-row"><span>This suburb</span><div class="track"><span style="width:{{ s.rate/maxrate*100 }}%"></span></div><b>{{ pct(s.rate) }}</b></div>
    <div class="bar-row"><span>Cape Town</span><div class="track"><span class="h" style="width:{{ city.rate/maxrate*100 }}%"></span></div><b>{{ pct(city.rate) }}</b></div>
    <p class="note">Dwellings surveyed: formal {{ pct(s.formal) }} · backyard {{ pct(s.adi) }} · informal {{ pct(s.informal) }}</p>
    <div class="mix" aria-hidden="true"><span class="m1" style="width:{{ s.formal*100 }}%"></span><span class="m2" style="width:{{ s.adi*100 }}%"></span><span class="m3" style="width:{{ s.informal*100 }}%"></span></div>
    <div><a class="btn ghost" href="/?suburb={{ s.name|urlencode }}">Screen a household in {{ s.name }} →</a></div>
  </div></div>{% endfor %}</div>
{% if matches|length > 6 %}<p class="note">{{ matches|length - 6 }} more matches are in the table below.</p>{% endif %}
{% endif %}
<div class="card"><div class="card-h"><h2>{{ 'Matching suburbs' if q else 'Suburbs ranked by adult hunger' }}</h2><span>{{ 'all ' ~ rows|length if (q or show_all) else 'top 20 of ' ~ lookup.suburbs|length }} · <a href="/suburbs?all=1">show all</a></span></div>
<div class="card-b"><div class="tbl"><table><thead><tr><th>#</th><th class="l">suburb</th><th>households</th><th>adult hunger</th><th>flagged by model</th><th class="l">mostly</th></tr></thead><tbody>
{% for s in rows %}<tr class="{{ 'small' if s.households < 40 }}"><td>{{ s.rank }}</td><td class="l"><a href="/suburbs?q={{ s.name|urlencode }}">{{ s.name }}</a></td><td>{{ s.households }}</td><td><b>{{ pct(s.rate) }}</b></td><td>{{ pct(s.flagged) }}</td><td class="l">{{ s.mostly }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="note">Survey rates for one suburb rest on few households (grey rows: under 40), so treat small differences as noise. Use this view to plan outreach areas, not to judge individual households.</p></div></div>
{% endif %}

{# ====================================================== BATCH #}
{% if page == 'batch' %}
{% set has = batch is not none %}
<section class="kpis">
  <div class="kpi" style="--f:{{ 1 if has else 0 }}"><span class="lab">Rows scored</span><span class="val">{{ '{:,}'.format(batch|length) if has else 0 }}</span><span class="sub">{{ fname if has else 'no file loaded' }}</span><svg class="i" viewBox="0 0 24 24">{{ icons.batch|safe }}</svg></div>
  <div class="kpi" style="--f:{{ batch.flagged.mean() if has else 0 }}"><span class="lab">Flagged</span><span class="val">{{ '{:,}'.format(batch.flagged.sum()) if has else 0 }}</span><span class="sub">{{ pct(batch.flagged.mean()) if has else '0%' }} of rows</span><svg class="i" viewBox="0 0 24 24">{{ icons.target|safe }}</svg></div>
  <div class="kpi" style="--f:{{ batch.score.mean() if has else 0 }}"><span class="lab">Average score</span><span class="val">{{ pct(batch.score.mean()) if has else '0%' }}</span><span class="sub">median {{ pct(batch.score.median()) if has else '0%' }}</span><svg class="i" viewBox="0 0 24 24">{{ icons.gauge|safe }}</svg></div>
  <div class="kpi" style="--f:{{ (1 - batch_missing|length / nfeat) if has else 0 }}"><span class="lab">Columns matched</span><span class="val">{{ nfeat - batch_missing|length if has else 0 }}/{{ nfeat }}</span><span class="sub">missing ones use typical values</span><svg class="i" viewBox="0 0 24 24">{{ icons.curve|safe }}</svg></div>
</section>
<div class="grid">
  <div class="card">
    <div class="card-h"><h2>{{ 'Scored rows' if batch is not none else 'Upload a CSV' }}</h2><span>{{ 'first 25 shown · download for all' if batch is not none else 'survey column names · max 20,000 rows, 50 MB' }}</span></div>
    <div class="card-b">
    {% if batch is not none %}
      {% if batch_missing %}<p class="note">{{ batch_missing|length }} expected columns were missing and were filled with typical values: {{ batch_missing[:10]|join(', ') }}{% if batch_missing|length > 10 %} …{% endif %}</p>{% endif %}
      <p>{{ batch.flagged.sum() }} of {{ batch|length }} rows flagged.</p>
      <div class="tbl">{{ batch.head(25).to_html(index=False, classes='t', border=0)|safe }}</div>
    {% else %}
      <div class="empty">Choose a CSV on the right. Each row is scored with the saved pipeline and you get a score and a flag per row.</div>
    {% endif %}
    </div>
  </div>
  <div class="sticky" style="display:grid;gap:22px">
    <form class="card" method="post" action="/batch" enctype="multipart/form-data">
      <div class="card-h"><h2>Score a file</h2></div>
      <div class="card-b">
        <label class="f" for="file">CSV file<input id="file" type="file" name="file" accept=".csv"><small>Use the column names from the data dictionary. Do not upload files containing names or ID numbers.</small></label>
        <div class="chips"><button class="btn" type="submit" name="action" value="show">Score and show</button><button class="btn ghost" type="submit" name="action" value="download">Score and download CSV</button></div>
        {% if has %}<div><a class="btn ghost" href="/batch">↺ Clear results</a></div><p class="note">Clears the scores and resets every count to 0, ready for the next file. Nothing is kept on the server.</p>{% endif %}
      </div>
    </form>
    <div class="card"><div class="card-h"><h2>Score distribution</h2><span>share of rows</span></div><div class="card-b"><div class="bars">
      {% for lab, share in (dist if has else empty_dist) %}<div class="bar-row"><span>{{ lab }}</span><div class="track">{% if share > 0 %}<span class="{{ 'h' if loop.index0 < 2 }}" style="width:{{ (share/(distmax or 1)*100)|round(1) }}%"></span>{% endif %}</div><b>{{ pct(share) }}</b></div>{% endfor %}
      </div><p class="note">Black bars are at or above the threshold (flagged).</p></div></div>
  </div>
</div>
{% endif %}

{# ====================================================== MODEL #}
{% if page == 'model' %}
<section class="kpis">
  <div class="kpi" style="--f:{{ ev.recall }}"><span class="lab">Recall</span><span class="val">{{ pct(ev.recall) }}</span><span class="sub">{{ (ev.recall*ev.test_pos)|round|int }} of {{ ev.test_pos }} at-risk found</span><svg class="i" viewBox="0 0 24 24">{{ icons.target|safe }}</svg></div>
  <div class="kpi" style="--f:{{ ev.precision }}"><span class="lab">Precision</span><span class="val">{{ pct(ev.precision) }}</span><span class="sub">of flags are truly at risk</span><svg class="i" viewBox="0 0 24 24">{{ icons.people|safe }}</svg></div>
  <div class="kpi" style="--f:{{ ev.macro_f1 }}"><span class="lab">Macro-F1</span><span class="val">{{ '%.3f'|format(ev.macro_f1) }}</span><span class="sub">accuracy {{ pct(ev.accuracy) }}</span><svg class="i" viewBox="0 0 24 24">{{ icons.model|safe }}</svg></div>
  <div class="kpi" style="--f:{{ ev.pr_auc }}"><span class="lab">PR-AUC</span><span class="val">{{ '%.3f'|format(ev.pr_auc) }}</span><span class="sub">ROC-AUC {{ '%.3f'|format(ev.roc_auc) }}</span><svg class="i" viewBox="0 0 24 24">{{ icons.curve|safe }}</svg></div>
</section>
<div class="grid even">
  <div class="card"><div class="card-h"><h2>Fairness by dwelling type</h2><span>locked test set, {{ '{:,}'.format(ev.test_n) }} households</span></div><div class="card-b">
    <div class="legend"><span><i></i>recall (at-risk found)</span><span><i class="h"></i>flagged rate</span></div>
    <div class="bars">{% for g in ev.fair %}
      <div class="bar-row"><span><b>{{ g[0] }}</b></span><div class="track"><span style="width:{{ g[4]*100 }}%"></span></div><b>{{ pct(g[4]) }}</b></div>
      <div class="bar-row"><span class="note">{{ '{:,}'.format(g[1]|int) }} households</span><div class="track"><span class="h" style="width:{{ g[3]*100 }}%"></span></div><b>{{ pct(g[3]) }}</b></div>{% endfor %}</div>
    <div class="tbl"><table><thead><tr><th>group</th><th>households</th><th>at risk</th><th>flagged</th><th>recall</th><th>precision</th></tr></thead><tbody>
      {% for g in ev.fair %}<tr><td>{{ g[0] }}</td><td>{{ '{:,}'.format(g[1]|int) }}</td><td>{{ pct(g[2]) }}</td><td>{{ pct(g[3]) }}</td><td>{{ pct(g[4]) }}</td><td>{{ pct(g[5]) }}</td></tr>{% endfor %}</tbody></table></div>
    <p class="note">The model misses more at-risk households in <b>formal dwellings</b> (recall {{ pct(ev.fair[0][4]) }}) than in informal settlements ({{ pct(ev.fair[2][4]) }}), and flags most informal-settlement households. Outreach teams should not treat "not flagged" in formal areas as safe.</p>
  </div></div>
  <div style="display:grid;gap:22px">
    <div class="card"><div class="card-h"><h2>McNemar test vs {{ ev.mcnemar.comparator }}</h2><span>same {{ '{:,}'.format(ev.test_n) }} test cases · α = 0.05</span></div><div class="card-b">
      <div class="mc"><span class="h"></span><span class="h">{{ ev.mcnemar.comparator }} correct</span><span class="h">{{ ev.mcnemar.comparator }} wrong</span>
        <span class="h" style="writing-mode:vertical-rl;transform:rotate(180deg)">MLP correct</span><div><b>{{ ev.mcnemar.both_correct }}</b>both correct</div><div><b>{{ ev.mcnemar.champion_only_correct }}</b>only MLP correct</div>
        <span class="h" style="writing-mode:vertical-rl;transform:rotate(180deg)">MLP wrong</span><div><b>{{ ev.mcnemar.comparator_only_correct }}</b>only LR correct</div><div><b>{{ ev.mcnemar.both_wrong }}</b>both wrong</div></div>
      <p class="note">p = {{ '%.3f'|format(ev.mcnemar.p) }} &gt; 0.05, so we cannot reject H₀: the tuned MLP and logistic regression make errors at a <b>statistically similar</b> rate on the same households.</p>
    </div></div>
    <div class="card"><div class="card-h"><h2>Do not use the model when…</h2></div><div class="card-b"><ul class="list">
      <li>Deciding <b>eligibility</b> for grants, food parcels or any benefit. It is a screening aid only.</li>
      <li>Households are <b>outside Cape Town</b> or the data is not from 2024. Conditions differ, so scores may not transfer.</li>
      <li>Most answers are unknown. Missing fields are filled with a "typical" household, which hides real need.</li>
      <li>Nobody reviews the flags. Every flag needs a follow-up conversation, and about 2 in 3 flags are not at risk.</li></ul>
      <p class="note"><b>Monitoring:</b> re-check recall by dwelling type each quarter, and retrain when a new survey is released.</p></div></div>
  </div>
</div>
{% endif %}

{# ====================================================== GUIDE #}
{% if page == 'guide' %}
<div class="grid even">
  <div class="card"><div class="card-h"><h2>Run it yourself</h2><span>locally reproducible launch</span></div><div class="card-b">
    <ol class="steps"><li><span>Install Python 3.12, open a terminal in the project folder, then create an environment and install the pinned packages:</span></li></ol>
    <pre>python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python wsgi.py</pre>
    <p class="note">On Mac/Linux activate with <code>source .venv/bin/activate</code>. <code>python app/app.py</code> also works (port 5000).</p>
    <ol class="steps" start="2" style="counter-reset:s 1"><li><span>Open <b>http://127.0.0.1:7860</b> (or :5000 with app/app.py). Stop with Ctrl+C.</span></li>
      <li><span>Hosted version: upload the Space folder to Hugging Face Spaces (Docker SDK). The <code>Dockerfile</code> runs <code>gunicorn wsgi:app</code> on port 7860.</span></li>
      <li><span>Reproduce the demonstration: <code>python app/demo_cases.py</code> writes the 3 valid + 1 invalid cases to <code>outputs/demo/</code>.</span></li></ol>
  </div></div>
  <div class="card"><div class="card-h"><h2>Demonstration cases</h2><span>run live through the saved pipeline</span></div><div class="card-b">
    <div class="tbl"><table><thead><tr><th class="l">case</th><th class="l">type</th><th>score</th><th class="l">outcome</th></tr></thead><tbody>
      {% for i, name, kind, out in demo_rows %}<tr><td class="l"><a href="/?demo={{i}}">{{i}}. {{ name }}</a></td><td class="l">{{ kind }}</td><td>{{ out[0] }}</td><td class="l" style="white-space:normal">{{ out[1] }}</td></tr>{% endfor %}</tbody></table></div>
    <p class="note">Click a case to open it in the screening form. The invalid case shows the validation messages instead of a score.</p></div></div>
  <div class="card"><div class="card-h"><h2>Input validation</h2></div><div class="card-b"><ul class="list">
    <li>Every drop-down and search box only accepts answers that appeared in the survey. Typed text is matched ignoring case, so &quot;delft&quot; finds Delft.</li>
    <li>The quick screen asks 9 questions; the rest are set to the most common answers for that dwelling type and listed with the result.</li>
    <li>Household size 1–20. Children must leave at least one adult. Grants cannot exceed household size.</li>
    <li>Spending must be a whole number from R0 to R200 000.</li>
    <li>"No income source" cannot be ticked together with another source, and income band "No income" cannot be combined with wages.</li>
    <li>Batch upload: only .csv files up to 50 MB and 20,000 rows. Unknown columns are ignored and missing ones are reported.</li></ul></div></div>
  <div class="card"><div class="card-h"><h2>Privacy and data</h2></div><div class="card-b"><ul class="list">
    <li>The app stores nothing. Inputs live only for the request that scores them.</li>
    <li>No credentials, keys or personal records are in the code or the Space.</li>
    <li>The raw survey data is not redistributed. Only the trained pipeline is included.</li>
    <li>Data: City of Cape Town, Household Survey 2024, Open Data Portal.</li></ul></div></div>
</div>
{% endif %}

</main></div></body></html>
"""

NAV = [("screen", "/", "Screen a household"), ("suburbs", "/suburbs", "Suburb lookup"), ("batch", "/batch", "Batch scoring"),
       ("model", "/model", "Model & evidence"), ("guide", "/guide", "Guide & deploy")]
SUBTITLES = {"screen": "Answer 9 quick questions (or the full form) to get a screening score, flag and explanation.",
             "suburbs": "Search any surveyed suburb to see how common adult hunger is there and how many households the model flags.",
             "batch": "Score many households at once from a CSV with the survey's column names.",
             "model": "How well the saved model performs on households it never saw, and where it falls short.",
             "guide": "How to launch the tool, how inputs are checked, and the demonstration cases."}
TITLES = {"screen": "Household Screening", "suburbs": "Suburb Lookup", "batch": "Batch Scoring", "model": "Model & Evidence", "guide": "Guide & Deploy"}


def pct(v, d=0):
    return f"{v * 100:.{d}f}%"


def render(page, form=None, result=None, errors=None, batch=None, batch_missing=None, demo=None, mode="quick", **extra):
    why, wmax = [], 1
    if result:
        why = sorted(result["effects"].items(), key=lambda kv: -abs(kv[1]))[:6]
        wmax = max([abs(e) for _, e in why] + [0.01])
    ctx = dict(page=page, title=TITLES[page], subtitle=SUBTITLES[page], nav=NAV, icons=ICONS, thr=THR, ev=EVID, pct=pct,
               form=form or {}, result=result, errors=errors, batch=batch, batch_missing=batch_missing or [], demo=demo,
               why=why, wmax=wmax, opt=options, selects=SELECTS, guide=GUIDE, income=INCOME_SOURCES, grants=GRANTS,
               assets=ASSET_COLS, asset_labels=ASSET_LABELS, shocks=SHOCKS, spending=SPENDING,
               demos=list(enumerate(DEMOS, 1)), nfeat=len(RAW_FEATURES), mode=mode, search_fields=SEARCH_FIELDS,
               quick_assets=QUICK_ASSETS, quick_income=QUICK_INCOME, lookup=LOOKUP, city=LOOKUP["city"],
               assumed=None, assumed_dt=None, sub_ctx=None)
    ctx.update(extra)
    return render_template_string(PAGE, **ctx)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        d = request.args.get("demo", type=int)
        if d and 1 <= d <= len(DEMOS):
            form = dict(DEMOS[d - 1][2])
            rec, errors = validate(form)
            if errors:
                return render("screen", form=form, errors=errors, demo=d, mode="full"), 400
            return render("screen", form=form, result=predict(rec), demo=d, mode="full", sub_ctx=SUBURBS.get(form.get("suburb")))
        mode = "full" if request.args.get("mode") == "full" else "quick"
        pre = {"suburb": canon("suburb", request.args.get("suburb", ""))} if request.args.get("suburb") else {}
        return render("screen", form=pre, mode=mode)
    form = request.form.to_dict()
    mode = "quick" if form.pop("mode", "full") == "quick" else "full"   # no mode field (API, demo script) = full form
    for f in SEARCH_FIELDS:                       # typed text -> exact answer (case and spacing ignored)
        if f in form:
            form[f] = canon(f, form[f])
    assumed = dt = None
    record = form
    if mode == "quick":
        record, assumed, dt = expand_quick(form)
    rec, errors = validate(record)
    if errors:
        return render("screen", form=form, errors=errors, mode=mode), 400
    return render("screen", form=form, result=predict(rec), mode=mode, assumed=assumed, assumed_dt=dt,
                  sub_ctx=SUBURBS.get(rec.get("suburb")))


@app.route("/suburbs")
def suburbs():
    q = " ".join(request.args.get("q", "").split())
    show_all = bool(request.args.get("all"))
    ranked = []
    for i, s in enumerate(LOOKUP["suburbs"], 1):
        mostly = max([("formal", s["formal"]), ("backyard", s["adi"]), ("informal", s["informal"])], key=lambda kv: kv[1])[0]
        ranked.append(dict(s, rank=i, mostly=mostly))
    matches = []
    if q:
        exact = [s for s in ranked if s["name"].lower() == q.lower()]
        matches = exact + [s for s in ranked if q.lower() in s["name"].lower() and s not in exact]   # exact match first
    rows = matches if q else (ranked if show_all else ranked[:20])
    maxrate = max([s["rate"] for s in matches] + [LOOKUP["city"]["rate"], 0.01])
    return render("suburbs", q=q, matches=matches, rows=rows, show_all=show_all, maxrate=maxrate), (404 if q and not matches else 200)


DIST_LABELS = ["Low (< 30%)", f"Moderate (30–{THR * 100:.0f}%)", f"Elevated ({THR * 100:.0f}–75%)", "High (≥ 75%)"]


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":          # empty state: every count at 0, ready for a file
        return render("batch", empty_dist=[(l, 0) for l in DIST_LABELS], distmax=1)
    f = request.files.get("file")
    if not f or not f.filename:
        return render("batch", errors=["Choose a CSV file to upload."], empty_dist=[(l, 0) for l in DIST_LABELS], distmax=1), 400
    if not f.filename.lower().endswith(".csv"):
        return render("batch", errors=["Only .csv files are accepted."], empty_dist=[(l, 0) for l in DIST_LABELS], distmax=1), 400
    try:
        df = pd.read_csv(io.BytesIO(f.read()), low_memory=False)
        out, missing = predict_csv(df)
    except Exception as e:   # show a helpful message instead of a stack trace
        return render("batch", errors=[f"Could not score this file: {e}"], empty_dist=[(l, 0) for l in DIST_LABELS], distmax=1), 400
    if request.form.get("action") == "download":
        full = df.copy()
        full["screening_score"] = out["score"].values
        full["flagged"] = out["flagged"].values
        return Response(full.to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=scored_households.csv"})
    edges = [0, 0.3, THR, 0.75, 1.0001]
    labels = DIST_LABELS
    shares = [((out.score >= a) & (out.score < b)).mean() for a, b in zip(edges[:-1], edges[1:])]
    return render("batch", batch=out, batch_missing=missing, fname=f.filename, dist=list(zip(labels, shares)),
                  distmax=max(shares + [0.01]))


@app.route("/model")
def model():
    return render("model")


@app.route("/guide")
def guide():
    rows = []
    for i, (name, kind, form) in enumerate(DEMOS, 1):
        rec, errors = validate(dict(form))
        if errors:
            rows.append((i, name, kind, ("–", f"{len(errors)} error messages shown, no score")))
        else:
            r = predict(rec)
            rows.append((i, name, kind, (pct(r["probability"]), ("Flagged" if r["flagged"] else "Not flagged") + f" · {r['band']}")))
    return render("guide", demo_rows=rows)


@app.errorhandler(413)
def too_large(_):
    return render("batch", errors=["File too large (limit 50 MB)."], empty_dist=[(l, 0) for l in DIST_LABELS], distmax=1), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)