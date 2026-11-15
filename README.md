# BDALakes

# NOTES 
BDA Lakes what more do you need

Objectives: 
- Model Comparison 
- Develop lake drawing Finland GUI with prediction of when, user chooses, vs actually when it does
- Relate it all to this Bayes guy

# REPORT

Lake ice break-up in Finland: BDA project

We model the day the ice breaks up on Finnish lakes from 1900 to 2026 with hierarchical Bayesian models in Bambi and PyMC. The data are SYKE ice observations (code 28, break-up of the ice cover) for the 64 lake sites whose records reach 2015 or later: 6,093 lake-years over 127 seasons.

Research questions:

RQ1: How does a lake's location affect when its ice breaks up?
RQ2: How has the break-up date changed since 1900, and has the change sped up since about 1980?
Main results

All numbers come from the preferred model, M3, as posterior means with 90% intervals.

Question	Result
H1: later break-up further north	Supported. 4.5 days later per degree north [4.2, 4.8], or about 4 days per 100 km. P(effect > 0) = 1.00
H1: longitude has little effect	Supported. 0.05 days per degree east [-0.27, 0.35]; the whole posterior lies inside the ±1 day/degree ROPE
H2: change has sped up since 1980	Supported. Trend before 1980 was -0.4 days/decade [-0.9, 0.05], close to flat. After 1980 it is -2.7 days/decade [-3.5, -1.8]. P(steeper after 1980) = 0.999
H2: total change	Ice breaks up about 12 days earlier than in 1980 [8, 16 days]
H3: lakes differ in their trend	Partly supported. Lake trends do differ (SD 0.24 days/decade [0.19, 0.29]), but much of the difference follows latitude: each degree north makes the trend 0.10 days/decade weaker [0.06, 0.15]. Southern lakes are changing fastest, roughly -2.9 days/decade at 61° N against -2.1 at 68.5° N

Other findings:

Year-to-year weather is the biggest source of variation. A nationwide year effect has SD 6.5 days, and regional deviations from it add another 3.9 days. Lakes at the same latitude differ by about 3 days (SD), and the remaining lake-year noise has scale 2.1 days.
The residuals have very heavy tails (Student-t ν ≈ 3.3), so a normal likelihood would have been a poor choice.
The spline model (M3s) has the same predictive performance as the 1980 broken stick (elpd difference 1.3, SE 0.8). Fixing the break at 1980 is therefore a fair summary of the curve's shape.

Projection, assuming the post-1980 trend continues (M3, typical lake, 90% predictive interval including weather):

Latitude	2030	2040
South (61° N)	18 Apr (3 Apr to 3 May)	16 Apr (31 Mar to 1 May)
Central (63.5° N)	1 May (15 Apr to 15 May)	28 Apr (13 Apr to 13 May)
North (68.5° N)	25 May (10 May to 8 Jun)	23 May (8 May to 7 Jun)

This is an extrapolation of a linear trend. Year-to-year weather (an interval about a month wide) matters much more than the 2-3 day shift per decade.

Models
Model	Formula (Bambi)	Purpose
M0 pooled	breakup_doy ~ lat_c + lon_c + decade_c	Baseline
M1 hierarchical, linear	... + lat_c:decade_c + (1 + decade_c | site) + (1 | season_f)	Lake levels and trends, shared year effect
M2 hierarchical, broken stick	M1 + post1980	Trend may change after 1980 (H2)
M3 regional years	M2 + (1 | season_region)	Year effects differ between south, central and north
M3s spline (extension)	M3 with a B-spline of year	Checks the 1980 breakpoint

Every model has a Student-t likelihood and explicit, proper, weakly informative priors (PRIOR_SPEC in icebreak_common.py, explained in notebook 02). In the prior predictive check, 95% of simulated dates fall between day 73 and 188, and under 1% fall outside February-July.

Model checking and comparison

Posterior predictive checks. M3 is the only model that reproduces how weakly the regions' yearly anomalies co-move: the observed south-north correlation is 0.54. M1 and M2 share one year effect across the country and almost always replicate higher correlations (p ≈ 1.0), so they fail this check. M0 has no year effects and gets only 2 of 13 decade means inside its 90% band; the hierarchical models get 12 of 13. Within-site SD by region is reproduced by all hierarchical models.

The decade-mean check does not separate M1 from M2. The year effects absorb the shape of the trend in the fitted years, so M1 fits past decades as well as M2.

Model	PSIS-LOO elpd (SE)	Diff to M3 (SE)	Leave-future-out elpd 2006-2026	Diff to M3 (SE)
M3 regional years	-15,393 (80)	0	-4,100	0
M2 broken stick	-16,890 (85)	-1,497 (66)	-4,150	-50 (4)
M1 linear	-16,890 (85)	-1,497 (66)	-4,422	-322 (19)
M0 pooled	-21,648 (59)	-6,256 (85)	-4,565	-465 (28)

M3 is clearly best on both criteria. PSIS-LOO cannot tell M1 and M2 apart: when one lake-year is left out, the other lakes that year reveal its weather. Predicting unseen future years is different, and there the broken stick (M2) beats the straight line (M1) by 322 elpd (SE 19). That is the strongest predictive evidence for H2. All Pareto k values are below 0.7.

Prior sensitivity. Refitting with every prior scale halved or doubled barely moves the estimates; the largest change is in the post-1980 slope of M2 (-1.9 to -2.1 days/decade), well inside its interval. Power-scaling sensitivity of the prior is below 0.05 for every parameter except ν in the pooled model M0 (0.11). The conclusions are therefore driven by the data.

Known issues to fix or report
M1 and M2 mixed slowly with 1000 draws per chain (max R-hat 1.10 for M1, 1.02 for M2). Notebook 02 now refits them with 2000 warm-up and 2000 sampling iterations. M3, which all results use, converged well (R-hat 1.00, min bulk ESS 1,147), and none of the fits had a divergent transition. If M1 or M2 still mixes slowly after the refit, report it; their LOO numbers are only used for comparison.
A few sensitivity and leave-future-out fits sit at R-hat 1.01 or 1.02, just over the strict threshold. Mention this in the convergence section.
The optional all-sites robustness cell at the end of notebook 03 stopped with "Not enough samples to build a trace", which usually means the sampling was interrupted. Rerun it or leave it out.
Files
icebreak_bayes/
├── data/                      icebreak_dataDEF.csv, icebreak_data_placesDEF.csv
├── icebreak_common.py         data subset, formulas, priors, helpers
├── 01_data_eda.ipynb          data description, figures 1-5
├── 02_models_fit.ipynb        models, priors, prior predictive, MCMC (saves results/*.nc)
├── 03_checks_results.ipynb    diagnostics, PPC, sensitivity, LOO/LFO, H1-H3 results
├── figures/                   figures used in the report
└── results/                   fitted models (.nc) and result tables (.csv)

Run the notebooks in order (01, 02, 03) after pip install -r requirements.txt. Notebook 02 runs 17 fits plus the longer refits of M1 and M2, about 1.5-2 hours in total on a laptop; each finished fit is saved and loaded instead of refitted on the next run. We used bambi 0.15.0, pymc 5.25.1 and arviz 0.23.4.

Report section	Where
Introduction, data	01
Models, priors, code, inference details	02
Convergence diagnostics	03 §1
Posterior predictive checks	03 §3
Prior sensitivity	02 (refits) and 03 §5
Model comparison	03 §4
Results and conclusion	03 §6-7, results/key_numbers.csv
AI use

We drafted parts of this code and text with an AI assistant (Claude). The course asks us to report this, and we have checked the code, results and text ourselves.