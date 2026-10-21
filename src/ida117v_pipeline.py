"""
IDA117V Group Project - shared pipeline code
Cape Town Household Survey 2024: predicting adult food insecurity.

This module is imported by the experiment notebook/script AND by the
user interface (Q7), so the saved pipeline can be loaded anywhere.
Everything here is either stateless (row-level cleaning) or a scikit-learn
compatible estimator/transformer that learns only from the data it is fit on.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42

# --------------------------------------------------------------------------
# 1. Column definitions
# --------------------------------------------------------------------------
TARGET_SOURCE = "hunger_adult"
TARGET_NA = "Not applicable (no adult in household)"

ID_ADMIN_COLS = ["ObjectId", "ObjectId2", "hhid", "column47", "consent",
                 "language_int", "date", "hhwgt"]

# Answered at the same time as the target and describing food hardship or its
# direct consequences -> excluded to prevent target leakage.
LEAKAGE_COLS = (["hunger_child", "without_food", "exp_food"] +
                ["spending_electricity", "spending_food", "spending_schoolfees",
                 "spending_none", "spending_rent", "spending_petrol",
                 "spending_debt", "spending_other"] +
                ["without_medical", "without_debtrepay", "without_shelter",
                 "without_rates", "without_fuel", "without_elec",
                 "without_airtime"])

IRRELEVANT_COLS = ["burial"]  # funeral preference: no plausible link to the task

ASSET_COLS = ["hhasset_fridge", "hhasset_stove", "hhasset_vacuum",
              "hhasset_washer", "hhasset_computer", "hhasset_sattv",
              "hhasset_dvd", "hhasset_vehicle", "hhasset_tv", "hhasset_radio",
              "hhasset_landline", "hhasset_cell", "hhasset_geyser",
              "hhasset_borehole", "hhasset_prepaidmeter",
              "hhasset_loadshedding", "hhasset_property",
              "hhasset_rental_short", "hhasset_rental_long"]

NEGEVENT_COLS = ["negevent_flooding", "negevent_fire", "negevent_crime_violent",
                 "negevent_hospital", "negevent_wind", "negevent_otherdamage",
                 "negevent_crime_nonviolent", "negevent_incomeloss",
                 "negevent_landloss", "negevent_other", "negevent_drought",
                 "negevent_hail", "negevent_landslide"]

INCOME_SRC_COLS = ["income_wages", "income_business", "income_remittance",
                   "income_pension", "income_grant", "income_farming",
                   "income_rental", "income_other", "income_none"]

GRANT_COLS = ["grants_oag", "grants_disability", "grants_csg", "grants_cdg",
              "grants_foster", "grants_warvet", "grants_inaid", "grants_covid"]

EXP_COLS = ["exp_energy", "exp_transport", "exp_acc", "exp_education",
            "exp_debt", "exp_saving", "exp_remittance"]

AMENITY_COLS = ["local_park", "local_library", "local_pool",
                "local_sportsground", "local_beach", "local_naturereserve",
                "local_hall", "city_park", "city_library", "city_pool",
                "city_sportsground", "city_beach", "city_naturereserve",
                "city_hall"]

DWELLING_CAT = ["dwellingtype", "material_walls", "material_roof", "tenure",
                "rental"]
SERVICE_CAT = ["water_piped", "water_drinking", "water_household",
               "water_outdoor", "water_interruption", "toilet", "hhwaste",
               "energy_cooking", "energy_lighting", "electricity",
               "electricity_grid", "internet"]
WORK_CAT = ["homebusiness", "homebusiness_nonres", "workfromhome_fulltime",
            "workfromhome_parttime", "homebusinesstype", "exp_none"]
LOCATION_CAT = ["suburb"]

HHINCOME_ORDER = ["No income", "R1 - R400", "R401 - R800", "R801 - R1 600",
                  "R1 601 - R3 200", "R3 201 - R6 400", "R6 401 - R12 800",
                  "R12 801 - R25 600", "R25 601 - R51 200",
                  "R51 201 - R102 400", "R102 401 - R204 800",
                  "R204 801 or more"]

VISITS_PER_MONTH = {"Never": 0.0, "Less than once a month": 0.5,
                    "Once a month": 1.0, "2-3 times a month": 2.5,
                    "Once a week": 4.0, "2-3 times a week": 10.0,
                    "4-5 times a week": 18.0, "Daily/Almost daily": 25.0}

# Engineered numeric features (created inside the pipeline)
ENGINEERED = ["asset_count", "negevent_count", "grants_total",
              "children_ratio", "hhincome_ord", "nonfood_exp_total",
              "service_deprivations"]

# Final model inputs, grouped (groups are reused for ablations in Q6)
FEATURE_GROUPS = {
    "location": LOCATION_CAT,
    "dwelling": DWELLING_CAT,
    "household": ["hhsize", "hhchildren"],
    "assets": ASSET_COLS,
    "services": SERVICE_CAT,
    "amenity_use": AMENITY_COLS,
    "shocks": NEGEVENT_COLS,
    "income_grants": INCOME_SRC_COLS + GRANT_COLS + ["hhincome"],
    "expenditure": EXP_COLS,
    "work": WORK_CAT,
}
RAW_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


# --------------------------------------------------------------------------
# 2. Stateless cleaning (row-level; safe before the split)
# --------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level fixes that learn nothing from the data."""
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("object").where(df[c].isna(), df[c].astype(str).str.strip())
    df = df.replace({"996": np.nan})          # survey skip code
    df.loc[df["city_hall"] == "10", "city_hall"] = np.nan   # invalid code
    df["homebusinesstype"] = df["homebusinesstype"].fillna("No home business")
    return df


def make_target(df: pd.DataFrame):
    """Drop households with no adult and return (X, y)."""
    df = df[df[TARGET_SOURCE] != TARGET_NA].copy()
    y = (df[TARGET_SOURCE] != "No, never").astype(int).rename("food_insecure")
    return df, y


# --------------------------------------------------------------------------
# 3. Feature engineering (stateless, runs as the first pipeline step)
# --------------------------------------------------------------------------
def engineer(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    yes = lambda cols: (X[cols] == "Yes").sum(axis=1)
    X["asset_count"] = yes(ASSET_COLS)
    X["negevent_count"] = yes(NEGEVENT_COLS)
    X["grants_total"] = X[GRANT_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    X["children_ratio"] = (pd.to_numeric(X["hhchildren"], errors="coerce") /
                           pd.to_numeric(X["hhsize"], errors="coerce").clip(lower=1))
    inc = X["hhincome"].astype(str).str.strip()
    X["hhincome_ord"] = inc.map({k: i for i, k in enumerate(HHINCOME_ORDER)})  # "Don't know" -> NaN
    X["nonfood_exp_total"] = X[EXP_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    piped = X["water_piped"].astype(str).str.contains("inside the dwelling|inside the yard", na=False)
    flush = X["toilet"].astype(str).str.startswith("Flush toilet connected", na=False)
    weekly = X["hhwaste"].astype(str).str.contains("at least once a week", na=False)
    elec = X["electricity"].astype(str).str.contains("In-house", na=False)
    X["service_deprivations"] = (~piped).astype(int) + (~flush) + (~weekly) + (~elec)
    for c in AMENITY_COLS:                      # ordinal visit frequency
        X[c] = X[c].map(VISITS_PER_MONTH)       # "Don't know"/"Not applicable" -> NaN
    return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise heavy-tailed rand amounts at a quantile learned on training data."""
    def __init__(self, q=0.99):
        self.q = q

    def fit(self, X, y=None):
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _cols(groups):
    feats = [c for g in groups for c in FEATURE_GROUPS[g]]
    cat = [c for c in feats if c in LOCATION_CAT + DWELLING_CAT + SERVICE_CAT +
           WORK_CAT + ASSET_COLS + NEGEVENT_COLS + INCOME_SRC_COLS]
    num = [c for c in feats if c in ["hhsize", "hhchildren"] + GRANT_COLS + AMENITY_COLS]
    exp = [c for c in feats if c in EXP_COLS]
    eng = []
    if "assets" in groups: eng.append("asset_count")
    if "shocks" in groups: eng.append("negevent_count")
    if "income_grants" in groups: eng += ["grants_total", "hhincome_ord"]
    if "household" in groups: eng.append("children_ratio")
    if "expenditure" in groups: eng.append("nonfood_exp_total")
    if "services" in groups: eng.append("service_deprivations")
    return cat, num + eng, exp


def make_preprocessor(kind="onehot", groups=None, scale=True, clip_log=True):
    """
    kind='onehot'  -> numeric + one-hot categoricals (traditional models, MLP)
    kind='ordinal' -> integer-coded categoricals FIRST, then numerics
                      (for the entity-embedding network)
    All imputers/scalers/encoders/clippers are fit on training folds only.
    """
    groups = list(FEATURE_GROUPS) if groups is None else groups
    cat, num, exp = _cols(groups)
    scaler = [("sc", StandardScaler())] if scale else []
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))] + scaler)
    if clip_log:
        exp_pipe = Pipeline([("clip", QuantileClipper(0.99)),
                             ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] + scaler)
    else:   # ablation: raw rand amounts, only scaled
        exp_pipe = Pipeline([("pass", FunctionTransformer(lambda a: np.asarray(a, dtype=float), feature_names_out="one-to-one"))] + scaler)
    if kind == "onehot":
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                             ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                  min_frequency=10, sparse_output=False))])
    else:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1, min_frequency=10))])
    transformers = [("cat", cat_pipe, cat), ("num", num_pipe, num)]
    if exp:
        transformers.append(("exp", exp_pipe, exp))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("engineer", FunctionTransformer(engineer)), ("columns", ct)]), len(cat)


# --------------------------------------------------------------------------
# 4. Neural model 1: MLP with per-epoch train/validation loss tracking
# --------------------------------------------------------------------------
class TrackedMLP(ClassifierMixin, BaseEstimator):
    """sklearn MLPClassifier trained epoch-by-epoch with class-balanced
    sample weights, early stopping on an internal stratified validation
    split (taken from the TRAINING data only) and best-epoch restore."""
    def __init__(self, hidden_layer_sizes=(128, 64), alpha=1e-3,
                 learning_rate_init=1e-3, batch_size=64, max_epochs=200,
                 patience=15, val_fraction=0.15, random_state=SEED,
                 class_weight="balanced", early_stopping=True):
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
        sss = StratifiedShuffleSplit(1, test_size=self.val_fraction, random_state=self.random_state)
        tr, va = next(sss.split(X, y))
        cw = getattr(self, "class_weight", "balanced")
        sw = compute_sample_weight(cw, y[tr]) if cw else np.ones(len(tr))
        swv = compute_sample_weight(cw, y[va]) if cw else np.ones(len(va))
        es = getattr(self, "early_stopping", True)
        rng = np.random.RandomState(self.random_state)
        net = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                            learning_rate_init=self.learning_rate_init,
                            batch_size=self.batch_size, solver="adam",
                            random_state=self.random_state)
        self.history_ = {"train_loss": [], "val_loss": [], "train_ap": [], "val_ap": []}
        best, best_loss, wait = None, np.inf, 0
        for epoch in range(self.max_epochs):
            idx = rng.permutation(len(tr))
            net.partial_fit(X[tr][idx], y[tr][idx], classes=[0, 1], sample_weight=sw[idx])
            p_tr = net.predict_proba(X[tr])[:, 1]; p_va = net.predict_proba(X[va])[:, 1]
            l_tr = log_loss(y[tr], p_tr, sample_weight=sw); l_va = log_loss(y[va], p_va, sample_weight=swv)
            self.history_["train_loss"].append(l_tr); self.history_["val_loss"].append(l_va)
            self.history_["train_ap"].append(average_precision_score(y[tr], p_tr))
            self.history_["val_ap"].append(average_precision_score(y[va], p_va))
            if l_va < best_loss - 1e-4:
                best, best_loss, wait = copy.deepcopy(net), l_va, 0
                self.best_epoch_ = epoch + 1
            else:
                wait += 1
                if es and wait >= self.patience:
                    break
        self.model_ = best if es else net
        if not es:
            self.best_epoch_ = len(self.history_["val_loss"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# 5. Neural model 2: entity-embedding tabular network (NumPy implementation)
#    Each categorical variable gets its own learned embedding vector
#    (Guo & Berkhahn, 2016), concatenated with scaled numeric features and
#    passed through two ReLU layers with dropout. Trained with Adam,
#    class-weighted binary cross-entropy, L2 weight decay and early stopping.
# --------------------------------------------------------------------------
class EmbeddingNetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_cat=0, max_emb_dim=8, hidden=(64, 32), dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=200,
                 patience=15, val_fraction=0.15, random_state=SEED):
        self.n_cat = n_cat
        self.max_emb_dim = max_emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ---- helpers ----
    def _split(self, X):
        X = np.asarray(X, dtype=float)
        codes = X[:, :self.n_cat].astype(int) + 1          # -1 (unknown) -> 0
        codes = np.clip(codes, 0, np.asarray(self.card_) - 1) if hasattr(self, "card_") else codes
        return codes, X[:, self.n_cat:]

    def _forward(self, codes, num, P, train, rng=None):
        embs = [P["E%d" % j][codes[:, j]] for j in range(self.n_cat)]
        z0 = np.concatenate(embs + [num], axis=1)
        cache = {"z0": z0, "codes": codes}
        h = z0
        for k in (1, 2):
            a = h @ P["W%d" % k] + P["b%d" % k]
            hr = np.maximum(a, 0)
            if train and self.dropout > 0:
                m = (rng.rand(*hr.shape) >= self.dropout) / (1 - self.dropout)
            else:
                m = np.ones_like(hr)
            cache["in%d" % k], cache["a%d" % k], cache["m%d" % k] = h, a, m
            h = hr * m
        cache["h2"] = h
        logit = h @ P["W3"] + P["b3"]
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30))), cache

    def _loss(self, p, y, sw):
        p = np.clip(p[:, 0], 1e-7, 1 - 1e-7)
        return float(np.sum(sw * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(sw))

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        Xa = np.asarray(X, dtype=float)
        self.card_ = [int(Xa[:, j].max()) + 2 for j in range(self.n_cat)]   # +1 shift, +1 unknown
        codes, num = self._split(Xa)
        rng = np.random.RandomState(self.random_state)
        sss = StratifiedShuffleSplit(1, test_size=self.val_fraction, random_state=self.random_state)
        tr, va = next(sss.split(Xa, y))
        sw_all = compute_sample_weight("balanced", y[tr])
        swv = compute_sample_weight("balanced", y[va])
        self.emb_dims_ = [min(self.max_emb_dim, (c + 1) // 2) for c in self.card_]
        d_in = sum(self.emb_dims_) + num.shape[1]
        h1, h2 = self.hidden
        P = {}
        for j, (c, d) in enumerate(zip(self.card_, self.emb_dims_)):
            P["E%d" % j] = rng.normal(0, 0.05, (c, d))
        P["W1"] = rng.normal(0, np.sqrt(2 / d_in), (d_in, h1)); P["b1"] = np.zeros(h1)
        P["W2"] = rng.normal(0, np.sqrt(2 / h1), (h1, h2)); P["b2"] = np.zeros(h2)
        P["W3"] = rng.normal(0, np.sqrt(1 / h2), (h2, 1)); P["b3"] = np.zeros(1)
        mom = {k: np.zeros_like(v) for k, v in P.items()}
        vel = {k: np.zeros_like(v) for k, v in P.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        self.history_ = {"train_loss": [], "val_loss": [], "train_ap": [], "val_ap": []}
        best, best_loss, wait = None, np.inf, 0
        for epoch in range(self.max_epochs):
            order = rng.permutation(len(tr))
            for s in range(0, len(order), self.batch_size):
                bi = tr[order[s:s + self.batch_size]]
                sw = sw_all[order[s:s + self.batch_size]]
                p, cache = self._forward(codes[bi], num[bi], P, True, rng)
                g = self._backward(p, y[bi], sw, cache, P)
                t += 1
                for k in P:
                    mom[k] = b1 * mom[k] + (1 - b1) * g[k]
                    vel[k] = b2 * vel[k] + (1 - b2) * g[k] ** 2
                    P[k] -= self.lr * (mom[k] / (1 - b1 ** t)) / (np.sqrt(vel[k] / (1 - b2 ** t)) + eps)
            p_tr, _ = self._forward(codes[tr], num[tr], P, False)
            p_va, _ = self._forward(codes[va], num[va], P, False)
            l_tr, l_va = self._loss(p_tr, y[tr], sw_all), self._loss(p_va, y[va], swv)
            self.history_["train_loss"].append(l_tr); self.history_["val_loss"].append(l_va)
            self.history_["train_ap"].append(average_precision_score(y[tr], p_tr[:, 0]))
            self.history_["val_ap"].append(average_precision_score(y[va], p_va[:, 0]))
            if l_va < best_loss - 1e-4:
                best, best_loss, wait = copy.deepcopy(P), l_va, 0
                self.best_epoch_ = epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.params_ = best
        self.classes_ = np.array([0, 1])
        return self

    def _backward(self, p, y, sw, cache, P):
        wd = self.weight_decay
        g = {}
        d = (sw * (p[:, 0] - y) / np.sum(sw))[:, None]            # dL/dlogit
        g["W3"] = cache["h2"].T @ d + wd * P["W3"]; g["b3"] = d.sum(0)
        dh = d @ P["W3"].T
        for k in (2, 1):
            da = dh * cache["m%d" % k] * (cache["a%d" % k] > 0)
            g["W%d" % k] = cache["in%d" % k].T @ da + wd * P["W%d" % k]
            g["b%d" % k] = da.sum(0)
            dh = da @ P["W%d" % k].T
        start = 0
        for j, dim in enumerate(self.emb_dims_):
            gE = np.zeros_like(P["E%d" % j])
            np.add.at(gE, cache["codes"][:, j], dh[:, start:start + dim])
            g["E%d" % j] = gE + wd * P["E%d" % j]
            start += dim
        return g

    def predict_proba(self, X):
        codes, num = self._split(X)
        p, _ = self._forward(codes, num, self.params_, False)
        return np.hstack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
