"""Data preparation and model definitions for our BDA lake-ice project.

All three notebooks import this file, so the data subset, predictors and
priors only need to be defined once.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# -- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
for _d in (FIG_DIR, RES_DIR):
    _d.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "icebreak_dataDEF.csv"
PLACES_FILE = DATA_DIR / "icebreak_data_placesDEF.csv"

# -- constants
LAT_CENTER = 62.98   # lat_c in the cleaned CSV is lat - 62.98 (mean over all 105 sites)
LON_CENTER = 26.0    # roughly the middle of Finland's east-west range
BREAK_YEAR = 1980    # knot of the broken-stick trend (H2)
REGION_CUTS = [0, 62.5, 65.0, 90]          # south / central / north
REGION_LABELS = ["south", "central", "north"]

RESPONSE = "breakup_doy"


def load_breakup(recent_only: bool = True, path=DATA_FILE) -> pd.DataFrame:
    """Rows with a valid break-up date (SYKE code 28), ready for modelling.

    recent_only=True keeps the 64 sites whose records reach 2015 or later
    (main analysis); False keeps all 105 lake sites (robustness check).
    """
    d = pd.read_csv(path)
    d = d[d[RESPONSE].notna() & ~d["breakup_suspicious"]].copy()
    if recent_only:
        d = d[d["reaches_recent"]]

    d["site"] = d["Paikka_Id"].astype(str)          # grouping factor (not Jarvi_Id)
    d["season_f"] = d["season"].astype(str)         # crossed year effect
    d["lon_c"] = d["lon"] - LON_CENTER
    # decade_c = (season - 1990) / 10 is already in the file
    d["post1980"] = np.clip(d["season"] - BREAK_YEAR, 0, None) / 10.0
    d["region"] = pd.cut(d["lat"], REGION_CUTS, labels=REGION_LABELS).astype(str)
    d["season_region"] = d["season_f"] + "_" + d["region"]
    return d.reset_index(drop=True)


# -- models
# Bambi fits group-specific terms as independent (no intercept-slope
# correlation). We mention this in the report.
FORMULAS = {
    "M0_pooled": "breakup_doy ~ lat_c + lon_c + decade_c",
    "M1_hier_linear": (
        "breakup_doy ~ lat_c + lon_c + decade_c + lat_c:decade_c"
        " + (1 + decade_c | site) + (1 | season_f)"
    ),
    "M2_hier_broken": (
        "breakup_doy ~ lat_c + lon_c + decade_c + post1980 + lat_c:decade_c"
        " + (1 + decade_c | site) + (1 | season_f)"
    ),
    "M3_hier_broken_region": (
        "breakup_doy ~ lat_c + lon_c + decade_c + post1980 + lat_c:decade_c"
        " + (1 + decade_c | site) + (1 | season_f) + (1 | season_region)"
    ),
}

# Priors on the day-of-year scale. Notebook 02 explains the choices.
PRIOR_SPEC = {
    # name: (distribution, mean, scale); all priors are proper
    "Intercept":      ("Normal", 130.0, 15.0),   # mid-May at lat 62.98, lon 26, year 1990
    "lat_c":          ("Normal", 0.0, 5.0),      # days per degree north
    "lon_c":          ("Normal", 0.0, 3.0),      # days per degree east
    "decade_c":       ("Normal", 0.0, 2.0),      # days per decade (pre-1980 slope in M2/M3)
    "post1980":       ("Normal", 0.0, 3.0),      # change in slope after 1980, days per decade
    "lat_c:decade_c": ("Normal", 0.0, 0.5),      # trend difference per degree latitude
    "1|site":         ("HalfNormal", None, 10.0),   # sd of lake intercepts (days)
    "decade_c|site":  ("HalfNormal", None, 1.0),    # sd of lake-specific trends (days/decade)
    "1|season_f":     ("HalfNormal", None, 10.0),   # sd of nationwide year shocks (days)
    "1|season_region": ("HalfNormal", None, 5.0),   # sd of regional year shocks (days)
    "sigma":          ("HalfNormal", None, 10.0),   # residual scale (days)
    "nu":             ("Gamma", 2.0, 0.1),          # Student-t dof (alpha, beta); mean 20
}


def make_priors(formula: str, scale: float = 1.0) -> dict:
    """Bambi prior dict for the terms that appear in `formula`.

    `scale` multiplies every prior scale (not the means). Used for the prior
    sensitivity refits (scale = 0.5 and 2).
    """
    import bambi as bmb

    pri = {}
    for name, (dist, a, b) in PRIOR_SPEC.items():
        if name == "nu":
            # scale the dof prior through its rate: mean 20*scale
            pri[name] = bmb.Prior("Gamma", alpha=a, beta=b / scale)
            continue
        if name == "sigma":
            pri[name] = bmb.Prior("HalfNormal", sigma=b * scale)
            continue
        if "|" in name:
            if name not in _group_terms(formula):
                continue
            pri[name] = bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=b * scale)
            )
        else:
            term_present = name == "Intercept" or _has_common_term(formula, name)
            if term_present:
                pri[name] = bmb.Prior("Normal", mu=a, sigma=b * scale)
    return pri


def _group_terms(formula: str) -> set:
    """'(1 + decade_c | site)' -> {'1|site', 'decade_c|site'} (Bambi's naming)."""
    import re

    out = set()
    for expr, grp in re.findall(r"\(([^|()]+)\|([^()]+)\)", formula.replace(" ", "")):
        for p in expr.split("+"):
            out.add(f"{p}|{grp}")
    return out


def _has_common_term(formula: str, term: str) -> bool:
    rhs = formula.split("~", 1)[1].replace(" ", "")
    parts = [p for p in rhs.split("+") if "|" not in p]
    return term in parts


def build_model(name: str, data: pd.DataFrame, prior_scale: float = 1.0,
                family: str = "t", formula: str | None = None):
    """Build (but do not fit) a Bambi model."""
    import bambi as bmb

    f = formula or FORMULAS[name]
    pri = make_priors(f, prior_scale)
    if family != "t":
        pri.pop("nu", None)
    # center_predictors=False: our predictors are already centred by hand, so the
    # Intercept prior refers to lat 62.98 N, lon 26 E, year 1990.
    try:
        return bmb.Model(f, data, family=family, priors=pri, center_predictors=False)
    except TypeError:  # very old Bambi without the argument
        import warnings
        warnings.warn("This Bambi version has no center_predictors argument; "
                      "upgrade to bambi>=0.13 so the Intercept prior keeps its meaning.")
        return bmb.Model(f, data, family=family, priors=pri)


SAMPLE_KW = dict(draws=1000, tune=1000, chains=4, cores=4,
                 target_accept=0.95, random_seed=2026)


# -- spline variant
SPLINE_DF = 8


def add_spline_basis(d: pd.DataFrame, df: int = SPLINE_DF, lo=1900, hi=2026) -> list:
    """Add cubic B-spline basis columns s1..s{df-1} of `season` to d (in place).

    The first basis function is dropped so the basis is not collinear with the
    intercept. Knots are fixed (lo, hi), so the same basis can be rebuilt for any
    subset of the data.
    """
    from scipy.interpolate import BSpline

    k = 3
    n_inner = df - k + 1
    inner = np.linspace(lo, hi, n_inner)
    t = np.r_[[lo] * k, inner, [hi] * k]
    x = np.clip(d["season"].to_numpy(dtype=float), lo, hi)
    B = BSpline.design_matrix(x, t, k).toarray()
    names = []
    for j in range(1, B.shape[1]):
        nm = f"s{j}"
        d[nm] = B[:, j]
        names.append(nm)
    return names


def spline_formula(names: list) -> str:
    return ("breakup_doy ~ lat_c + lon_c + " + " + ".join(names)
            + " + lat_c:decade_c + (1 + decade_c | site) + (1 | season_f) + (1 | season_region)")


def spline_priors(names: list, scale: float = 1.0) -> dict:
    """Priors for the spline model: same as M3, plus N(0, 10) on each basis
    coefficient (deviation in days from the 1900 level)."""
    import bambi as bmb

    f = spline_formula(names)
    pri = make_priors(f, scale)
    for nm in names:
        pri[nm] = bmb.Prior("Normal", mu=0, sigma=10.0 * scale)
    return pri


# -- fitting helpers
def fit_or_load(model, path, force=False, **kw):
    """Fit a Bambi model and save the InferenceData, or load it if it exists."""
    import arviz as az

    path = Path(path)
    if path.exists() and not force:
        return az.from_netcdf(path)
    skw = {**SAMPLE_KW, **kw}
    # omit_offsets=False keeps the non-centred offsets, which the power-scaling
    # sensitivity check (compute_log_prior) needs.
    try:
        # forkserver is more reliable than the default start method on macOS
        idata = model.fit(idata_kwargs={"log_likelihood": True}, omit_offsets=False,
                          mp_ctx="forkserver", **skw)
    except (EOFError, BrokenPipeError, RuntimeError) as e:
        # a worker process died; run the chains one after another instead
        import warnings
        warnings.warn(f"Parallel sampling failed ({type(e).__name__}); retrying with cores=1.")
        skw["cores"] = 1
        idata = model.fit(idata_kwargs={"log_likelihood": True}, omit_offsets=False, **skw)
    if "log_likelihood" not in idata.groups():
        model.compute_log_likelihood(idata)
    idata.to_netcdf(path)
    return idata


def predict(model, idata, kind, **kw):
    """model.predict with the kind names of both new and old Bambi versions."""
    alias = {"response": "pps", "response_params": "mean"}
    try:
        return model.predict(idata, kind=kind, **kw)
    except (ValueError, TypeError, KeyError):
        return model.predict(idata, kind=alias[kind], **kw)


def pvar(post, name):
    """Posterior variable by name, also accepting older Bambi naming."""
    for cand in (name, f"{RESPONSE}_{name}"):
        if cand in post:
            return post[cand]
    raise KeyError(name)


def draws(post, name):
    """Flatten chains: array of shape (S,) or (S, levels)."""
    x = pvar(post, name).stack(sample=("chain", "draw"))
    return x.transpose("sample", ...).values


def mu_draws(pred_idata):
    """(S, N) array of the linear predictor from a response_params prediction."""
    post = pred_idata.posterior
    for cand in ("mu", f"{RESPONSE}_mean"):
        if cand in post:
            x = post[cand].stack(sample=("chain", "draw"))
            return x.transpose("sample", ...).values
    raise KeyError("No mean variable found in prediction")


def pointwise_lpd(model, idata, newdata, family="t"):
    """log p(y_new | y_train) for each new observation (Student-t or normal).

    Group effects for unseen seasons (and season-region cells) are drawn from
    their fitted population distribution, so the predictive uncertainty
    includes ordinary year-to-year weather.
    """
    from scipy import stats
    from scipy.special import logsumexp

    pred = predict(model, idata, "response_params", data=newdata,
                   sample_new_groups=True, inplace=False)
    mu = mu_draws(pred)
    sig = draws(idata.posterior, "sigma")[:, None]
    y = newdata[RESPONSE].to_numpy()[None, :]
    if family == "t":
        nu = draws(idata.posterior, "nu")[:, None]
        ll = stats.t.logpdf(y, df=nu, loc=mu, scale=sig)
    else:
        ll = stats.norm.logpdf(y, loc=mu, scale=sig)
    return logsumexp(ll, axis=0) - np.log(ll.shape[0])


def scalar_vars(idata):
    """Names of scalar posterior parameters (fixed effects, group SDs, sigma, nu)."""
    post = idata.posterior
    return [v for v in post.data_vars if set(post[v].dims) == {"chain", "draw"}]


def diagnostics_table(idata):
    """Max R-hat, min ESS (over all parameters) and number of divergences."""
    import arviz as az

    s = az.summary(idata, kind="diagnostics")
    return {"max R-hat": float(s["r_hat"].max()),
            "min bulk-ESS": float(s["ess_bulk"].min()),
            "min tail-ESS": float(s["ess_tail"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum()),
            "parameters": len(s)}