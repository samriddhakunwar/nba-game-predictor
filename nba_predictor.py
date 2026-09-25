"""
NBA Game Outcome Predictor
==========================

Predicts the winner of NBA regular-season games using only information that
was available BEFORE tip-off: team strength (Elo), season-to-date efficiency
and Four Factors, recent form (last 10 games), and rest / back-to-backs.

Pipeline
--------
1. Load team game logs (2010-11 to 2025-26) and pair each team with its opponent.
2. Compute per-game advanced stats (possessions, ratings, Four Factors).
3. Build PRE-GAME features for every team (strictly shifted, no leakage).
4. Turn each game into one row: home-team features minus away-team features.
5. Train baselines, logistic regression and gradient boosting.
6. Evaluate on a held-out season (2025-26) and with walk-forward validation.

Data: 2010-11 to 2023-24 from https://github.com/NocturneBear/NBA-Data-2010-2024
      (MIT licence, sourced from NBA.com); 2024-25 and 2025-26 from ESPN team box
      scores published by sportsdataverse. Run add_espn_seasons.py first to build
      the combined file.

Run:  python nba_predictor.py          (writes results to ./outputs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "regular_season_totals_2010_2026.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42

# Seasons ------------------------------------------------------------------
BURN_IN = "2010-11"      # used only to warm up Elo / prior-season stats
VALID_SEASON = "2024-25"  # used to choose model settings
TEST_SEASON = "2025-26"   # never touched until the final evaluation

# Elo settings (FiveThirtyEight-style) -------------------------------------
ELO_START = 1505
ELO_K = 20
ELO_HOME_ADV = 100
ELO_SEASON_CARRY = 0.75   # keep 75% of last season's rating, regress 25% to mean

# Shrinkage of season-to-date stats toward last season (in "games" of weight)
PRIOR_GAMES = 10
FORM_WINDOW = 10


# ---------------------------------------------------------------------------
# 1. Load + pair team rows with opponents
# ---------------------------------------------------------------------------
def load_team_games(path: Path = DATA) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = ["SEASON_YEAR", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
            "TEAM_NAME", "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A",
            "FTM", "FTA", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF"]
    df = raw[cols].copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"season_year": "season", "team_abbreviation": "team"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_home"] = df["matchup"].str.contains(" vs. ").astype(int)
    df["win"] = (df["wl"] == "W").astype(int)

    # Attach the opponent's box score to each team row
    stat_cols = ["team_id", "team", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                 "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]
    opp = df[["game_id"] + stat_cols].rename(columns={c: f"opp_{c}" for c in stat_cols})
    df = df.merge(opp, on="game_id")
    df = df[df["team_id"] != df["opp_team_id"]].copy()

    # One row per team per game -> exactly two rows per game
    assert df.groupby("game_id").size().eq(2).all()
    return df.sort_values(["game_date", "game_id", "is_home"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Per-game advanced stats
# ---------------------------------------------------------------------------
def add_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    team_poss = d.fga + 0.44 * d.fta - d.oreb + d.tov
    opp_poss = d.opp_fga + 0.44 * d.opp_fta - d.opp_oreb + d.opp_tov
    d["poss"] = (team_poss + opp_poss) / 2
    d["pace"] = d["poss"] * 48 / d["min"]
    d["ortg"] = 100 * d.pts / d.poss
    d["drtg"] = 100 * d.opp_pts / d.poss
    d["net_rtg"] = d.ortg - d.drtg
    d["margin"] = d.pts - d.opp_pts

    # Dean Oliver's Four Factors, for the team (offence) and allowed (defence)
    d["efg"] = (d.fgm + 0.5 * d.fg3m) / d.fga
    d["tov_pct"] = d.tov / d.poss
    d["orb_pct"] = d.oreb / (d.oreb + d.opp_dreb)
    d["ft_rate"] = d.ftm / d.fga
    d["opp_efg"] = (d.opp_fgm + 0.5 * d.opp_fg3m) / d.opp_fga
    d["opp_tov_pct"] = d.opp_tov / d.poss
    d["drb_pct"] = d.dreb / (d.dreb + d.opp_oreb)
    d["opp_ft_rate"] = d.opp_ftm / d.opp_fga
    d["fg3a_rate"] = d.fg3a / d.fga
    return d


# ---------------------------------------------------------------------------
# 3a. Elo ratings (computed game by game, pre-game value stored)
# ---------------------------------------------------------------------------
def elo_expected(diff: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + 10 ** (-diff / 400))


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per game with pre-game Elo for home and away teams."""
    games = (df[df.is_home == 1][["game_id", "game_date", "season", "team_id",
                                  "opp_team_id", "pts", "opp_pts"]]
             .sort_values(["game_date", "game_id"]))
    rating: dict[int, float] = {}
    last_season: dict[int, str] = {}
    rows = []
    for g in games.itertuples(index=False):
        for t in (g.team_id, g.opp_team_id):
            if t not in rating:
                rating[t] = ELO_START
            elif last_season[t] != g.season:  # new season -> regress to mean
                rating[t] = ELO_SEASON_CARRY * rating[t] + (1 - ELO_SEASON_CARRY) * ELO_START
            last_season[t] = g.season
        h, a = rating[g.team_id], rating[g.opp_team_id]
        rows.append((g.game_id, h, a))

        diff = h + ELO_HOME_ADV - a
        exp_home = elo_expected(diff)
        home_won = g.pts > g.opp_pts
        mov = abs(g.pts - g.opp_pts)
        winner_diff = diff if home_won else -diff
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        shift = ELO_K * mult * ((1 if home_won else 0) - exp_home)
        rating[g.team_id] += shift
        rating[g.opp_team_id] -= shift
    return pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo"])


# ---------------------------------------------------------------------------
# 3b. Pre-game team features (season-to-date, recent form, rest)
# ---------------------------------------------------------------------------
STRENGTH_STATS = ["win", "net_rtg", "ortg", "drtg", "margin", "pace",
                  "efg", "tov_pct", "orb_pct", "ft_rate",
                  "opp_efg", "opp_tov_pct", "drb_pct", "opp_ft_rate", "fg3a_rate"]


def build_team_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["team_id", "game_date"]).copy()
    g_season = d.groupby(["team_id", "season"], sort=False)

    # Games played before this one, this season
    d["gp"] = g_season.cumcount()

    # Previous season's full-season averages = the "prior" for early-season games
    season_avg = d.groupby(["team_id", "season"])[STRENGTH_STATS].mean().reset_index()
    seasons = sorted(d.season.unique())
    next_season = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    prior = season_avg.copy()
    prior["season"] = prior["season"].map(next_season)
    prior = prior.dropna(subset=["season"])
    league_avg = d.groupby("season")[STRENGTH_STATS].mean()

    # Regress last season 30% toward that season's league average
    # (rosters change over the summer, so last year is only partly informative)
    prior = prior.set_index(["team_id", "season"])
    for s in STRENGTH_STATS:
        league_prev = prior.index.get_level_values("season").map(
            lambda x: league_avg.loc[seasons[seasons.index(x) - 1], s])
        prior[s] = 0.7 * prior[s] + 0.3 * np.asarray(league_prev)
    prior = prior.add_prefix("prior_").reset_index()
    d = d.merge(prior, on=["team_id", "season"], how="left")
    # First season in the data (burn-in) has no prior -> use league average
    for s in STRENGTH_STATS:
        d[f"prior_{s}"] = d[f"prior_{s}"].fillna(d.season.map(league_avg[s]))

    d = d.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    g_season = d.groupby(["team_id", "season"], sort=False)

    for s in STRENGTH_STATS:
        # Mean of all PREVIOUS games this season (shift(1) = no leakage)
        prev_sum = g_season[s].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
        # Shrink toward the prior when only a few games have been played
        d[f"szn_{s}"] = (prev_sum + PRIOR_GAMES * d[f"prior_{s}"]) / (d["gp"] + PRIOR_GAMES)

    # Recent form: last 10 games this season (falls back to season value early on)
    for s in ["win", "net_rtg", "ortg", "drtg"]:
        roll = g_season[s].transform(
            lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=3).mean())
        d[f"form_{s}"] = roll.fillna(d[f"szn_{s}"])

    # Rest
    prev_date = g_season["game_date"].shift(1)
    d["rest_days"] = (d["game_date"] - prev_date).dt.days.fillna(7).clip(upper=7)
    d["b2b"] = (d["rest_days"] == 1).astype(int)
    # Games in the previous 7 days (fatigue from dense schedules)
    def games_in_prev_7_days(dates: pd.Series) -> np.ndarray:
        t = dates.values.astype("datetime64[D]")
        # number of earlier games whose date falls in [t-7 days, t)
        return np.arange(len(t)) - np.searchsorted(t, t - np.timedelta64(7, "D"), side="left")

    d["games_last7"] = g_season["game_date"].transform(games_in_prev_7_days)
    return d


# ---------------------------------------------------------------------------
# 4. One row per game: home minus away
# ---------------------------------------------------------------------------
def build_game_table(team_feats: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    feat_cols = ([f"szn_{s}" for s in STRENGTH_STATS]
                 + [f"form_{s}" for s in ["win", "net_rtg", "ortg", "drtg"]]
                 + ["rest_days", "b2b", "games_last7", "gp"])
    keep = ["game_id", "season", "game_date", "team_id", "team", "team_name",
            "pts", "win"] + feat_cols
    home = team_feats[team_feats.is_home == 1][keep].add_prefix("home_")
    away = team_feats[team_feats.is_home == 0][keep].add_prefix("away_")
    g = home.merge(away, left_on="home_game_id", right_on="away_game_id")
    g = g.rename(columns={"home_game_id": "game_id", "home_season": "season",
                          "home_game_date": "game_date"})
    g = g.drop(columns=["away_game_id", "away_season", "away_game_date", "away_win"])
    g = g.merge(elo, on="game_id")

    g["elo_diff"] = g.home_elo - g.away_elo
    g["elo_prob"] = elo_expected(g.elo_diff + ELO_HOME_ADV)
    for c in feat_cols:
        if c == "gp":
            continue
        g[f"diff_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


FEATURES = (["elo_diff"]
            + [f"diff_szn_{s}" for s in STRENGTH_STATS]
            + [f"diff_form_{s}" for s in ["win", "net_rtg", "ortg", "drtg"]]
            + ["home_rest_days", "away_rest_days", "home_b2b", "away_b2b",
               "diff_games_last7"])

FEATURE_LABELS = {
    "elo_diff": "Elo rating gap",
    "diff_szn_win": "Season win % gap",
    "diff_szn_net_rtg": "Season net rating gap",
    "diff_szn_ortg": "Season offensive rating gap",
    "diff_szn_drtg": "Season defensive rating gap",
    "diff_szn_margin": "Season point margin gap",
    "diff_szn_pace": "Pace gap",
    "diff_szn_efg": "Shooting (eFG%) gap",
    "diff_szn_tov_pct": "Turnover rate gap",
    "diff_szn_orb_pct": "Off. rebounding gap",
    "diff_szn_ft_rate": "Free-throw rate gap",
    "diff_szn_opp_efg": "Opp. shooting allowed gap",
    "diff_szn_opp_tov_pct": "Forced turnovers gap",
    "diff_szn_drb_pct": "Def. rebounding gap",
    "diff_szn_opp_ft_rate": "Opp. FT rate allowed gap",
    "diff_szn_fg3a_rate": "3-point attempt rate gap",
    "diff_form_win": "Last-10 win % gap",
    "diff_form_net_rtg": "Last-10 net rating gap",
    "diff_form_ortg": "Last-10 offence gap",
    "diff_form_drtg": "Last-10 defence gap",
    "home_rest_days": "Home rest days",
    "away_rest_days": "Away rest days",
    "home_b2b": "Home on back-to-back",
    "away_b2b": "Away on back-to-back",
    "diff_games_last7": "Schedule density gap (games in 7 days)",
}


# ---------------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------------
def make_logreg(C: float = 0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def make_gbm(lr: float = 0.03, depth: int = 3, iters: int = 300, leaf: int = 50):
    return HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth,
                                          max_iter=iters, min_samples_leaf=leaf,
                                          l2_regularization=1.0, random_state=SEED)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": accuracy_score(y, p >= 0.5),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def baseline_probs(df: pd.DataFrame, home_rate: float) -> dict[str, np.ndarray]:
    """home_rate = share of home wins in the training data"""
    better = np.where(df.diff_szn_win > 0, 0.65,
                      np.where(df.diff_szn_win < 0, 0.35, home_rate))
    return {"Home team always wins": np.full(len(df), home_rate),
            "Better record wins": better,
            "Elo only": df.elo_prob.values}


def tune(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    X, y = train[FEATURES], train.home_win
    Xv, yv = valid[FEATURES], valid.home_win
    rows = []
    best_lr, best_lr_ll = None, 9
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        m = make_logreg(C).fit(X, y)
        ll = log_loss(yv, m.predict_proba(Xv)[:, 1])
        rows.append({"model": "Logistic regression", "params": f"C={C}", "valid_log_loss": ll,
                     "valid_accuracy": accuracy_score(yv, m.predict(Xv))})
        if ll < best_lr_ll:
            best_lr, best_lr_ll = {"C": C}, ll
    best_gb, best_gb_ll = None, 9
    for lr in [0.01, 0.02, 0.05]:
        for depth in [1, 2, 3, 4]:
            for iters in [100, 200, 400]:
                m = make_gbm(lr, depth, iters).fit(X, y)
                ll = log_loss(yv, m.predict_proba(Xv)[:, 1])
                rows.append({"model": "Gradient boosting",
                             "params": f"lr={lr}, depth={depth}, trees={iters}",
                             "valid_log_loss": ll,
                             "valid_accuracy": accuracy_score(yv, m.predict(Xv))})
                if ll < best_gb_ll:
                    best_gb, best_gb_ll = {"lr": lr, "depth": depth, "iters": iters}, ll
    return best_lr, best_gb, pd.DataFrame(rows)


def fit_predict_all(train: pd.DataFrame, test: pd.DataFrame,
                    lr_params: dict, gb_params: dict) -> tuple[dict, dict]:
    X, y = train[FEATURES], train.home_win
    lr = make_logreg(**lr_params).fit(X, y)
    gb = make_gbm(**gb_params).fit(X, y)
    probs = baseline_probs(test, home_rate=y.mean())
    probs["Logistic regression"] = lr.predict_proba(test[FEATURES])[:, 1]
    probs["Gradient boosting"] = gb.predict_proba(test[FEATURES])[:, 1]
    probs["Ensemble (LR + GB)"] = (probs["Logistic regression"] + probs["Gradient boosting"]) / 2
    return probs, {"lr": lr, "gb": gb}


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    tg = add_advanced_stats(load_team_games())
    elo = compute_elo(tg)
    team_feats = build_team_features(tg)
    games = build_game_table(team_feats, elo)
    games.to_csv(OUT / "game_features.csv", index=False, lineterminator="\n")

    seasons = sorted(games.season.unique())
    usable = games[games.season != BURN_IN]
    train = usable[usable.season < VALID_SEASON]
    valid = usable[usable.season == VALID_SEASON]
    test = usable[usable.season == TEST_SEASON]
    print(f"Games: total {len(games)}, train {len(train)}, valid {len(valid)}, test {len(test)}")

    # --- choose settings on the validation season only ---
    lr_params, gb_params, tuning = tune(train, valid)
    tuning.to_csv(OUT / "tuning_results.csv", index=False, lineterminator="\n")
    print("Best LR:", lr_params, " Best GB:", gb_params)

    # --- final: train on everything before the test season, evaluate once ---
    train_full = usable[usable.season < TEST_SEASON]
    probs, models = fit_predict_all(train_full, test, lr_params, gb_params)
    y = test.home_win.values
    results = []
    for name, p in probs.items():
        r = evaluate(y, p)
        r["model"] = name
        results.append(r)
    results = pd.DataFrame(results)[["model", "accuracy", "log_loss", "brier", "auc"]]
    print(f"\nHeld-out {TEST_SEASON} season\n", results.round(4).to_string(index=False))
    results.to_csv(OUT / "test_results.csv", index=False, lineterminator="\n")

    # Model used for the game-by-game analysis: pick the best log loss of the ML models
    ml = results[results.model.isin(["Logistic regression", "Gradient boosting",
                                     "Ensemble (LR + GB)"])]
    best_name = ml.sort_values("log_loss").iloc[0]["model"]
    p_best = probs[best_name]
    print("Best model:", best_name)

    # --- walk-forward validation: every season predicted by a model trained on prior seasons
    wf_rows = []
    for s in seasons[3:]:
        tr = usable[usable.season < s]
        te = usable[usable.season == s]
        pr, _ = fit_predict_all(tr, te, lr_params, gb_params)
        for name, p in pr.items():
            r = evaluate(te.home_win.values, p)
            wf_rows.append({"season": s, "model": name, "games": len(te), **r})
    walk = pd.DataFrame(wf_rows)
    walk.to_csv(OUT / "walk_forward.csv", index=False, lineterminator="\n")
    print("\nWalk-forward mean accuracy\n",
          walk.groupby("model").accuracy.mean().sort_values().round(4))

    # --- feature-group ablation: what does each block of features add? ---
    # (Coefficients are hard to read when features are correlated, so instead we
    #  add groups one at a time and measure walk-forward performance.)
    groups = [
        ("Elo rating only", ["elo_diff"]),
        ("+ Season strength", ["diff_szn_win", "diff_szn_net_rtg", "diff_szn_ortg",
                               "diff_szn_drtg", "diff_szn_margin"]),
        ("+ Four Factors & style", ["diff_szn_efg", "diff_szn_tov_pct", "diff_szn_orb_pct",
                                    "diff_szn_ft_rate", "diff_szn_opp_efg",
                                    "diff_szn_opp_tov_pct", "diff_szn_drb_pct",
                                    "diff_szn_opp_ft_rate", "diff_szn_fg3a_rate",
                                    "diff_szn_pace"]),
        ("+ Recent form (last 10)", ["diff_form_win", "diff_form_net_rtg", "diff_form_ortg",
                                     "diff_form_drtg"]),
        ("+ Rest & schedule", ["home_rest_days", "away_rest_days", "home_b2b", "away_b2b",
                               "diff_games_last7"]),
    ]
    assert sorted(sum((g for _, g in groups), [])) == sorted(FEATURES)
    abl_rows, cols = [], []
    for label, g in groups:
        cols = cols + g
        accs, lls, ns = [], [], []
        for s in seasons[3:]:
            tr, te = usable[usable.season < s], usable[usable.season == s]
            mdl = make_logreg(**lr_params).fit(tr[cols], tr.home_win)
            p = mdl.predict_proba(te[cols])[:, 1]
            accs.append(accuracy_score(te.home_win, p >= 0.5))
            lls.append(log_loss(te.home_win, p))
            ns.append(len(te))
        abl_rows.append({"step": label, "n_features": len(cols),
                         "accuracy": np.average(accs, weights=ns),
                         "log_loss": np.average(lls, weights=ns)})
    ablation = pd.DataFrame(abl_rows)
    ablation.to_csv(OUT / "feature_group_ablation.csv", index=False, lineterminator="\n")
    print("\nFeature-group ablation (walk-forward)\n", ablation.round(4).to_string(index=False))

    # --- descriptive: how much do back-to-backs matter? ---
    rest_rows = []
    for label, mask in [("Neither team on a back-to-back", (games.home_b2b == 0) & (games.away_b2b == 0)),
                        ("Only away team on a back-to-back", (games.home_b2b == 0) & (games.away_b2b == 1)),
                        ("Only home team on a back-to-back", (games.home_b2b == 1) & (games.away_b2b == 0)),
                        ("Both on a back-to-back", (games.home_b2b == 1) & (games.away_b2b == 1))]:
        sub = games[mask & (games.season != BURN_IN)]
        rest_rows.append({"situation": label, "games": len(sub),
                          "home_win_rate": sub.home_win.mean()})
    rest_split = pd.DataFrame(rest_rows)
    print("\n", rest_split.round(3).to_string(index=False))

    # --- feature importance ---
    lr_model = models["lr"]
    coefs = pd.Series(lr_model[-1].coef_[0], index=FEATURES)
    perm = permutation_importance(models["gb"], test[FEATURES], y, scoring="neg_log_loss",
                                  n_repeats=10, random_state=SEED)
    importance = pd.DataFrame({"feature": FEATURES,
                               "label": [FEATURE_LABELS[f] for f in FEATURES],
                               "lr_coef_std": coefs.values,
                               "gb_perm_importance": perm.importances_mean})
    importance.to_csv(OUT / "feature_importance.csv", index=False, lineterminator="\n")

    # --- game-level predictions for the test season ---
    pred = test[["game_id", "game_date", "home_team", "away_team", "home_team_name",
                 "away_team_name", "home_pts", "away_pts", "home_win",
                 "home_elo", "away_elo"]].copy()
    pred["p_home"] = p_best
    pred["pred_home_win"] = (pred.p_home >= 0.5).astype(int)
    pred["correct"] = (pred.pred_home_win == pred.home_win).astype(int)
    pred["confidence"] = np.maximum(pred.p_home, 1 - pred.p_home)
    pred["elo_p_home"] = probs["Elo only"]
    pred.to_csv(OUT / f"predictions_{TEST_SEASON.replace('-', '_')}.csv", index=False, lineterminator="\n")

    # Calibration
    bins = np.linspace(0, 1, 11)
    pred["bin"] = pd.cut(pred.p_home, bins, include_lowest=True)
    calib = (pred.groupby("bin", observed=True)
             .agg(pred_mean=("p_home", "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
             .reset_index(drop=True))

    # Accuracy by confidence
    conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["50–60%", "60–70%", "70–80%", "80–90%", "90%+"]
    pred["conf_bin"] = pd.cut(pred.confidence, conf_bins, labels=labels, include_lowest=True)
    by_conf = (pred.groupby("conf_bin", observed=False)
               .agg(games=("correct", "size"), accuracy=("correct", "mean"),
                    avg_conf=("confidence", "mean")).reset_index())

    # Accuracy by month
    pred["month"] = pred.game_date.dt.to_period("M").astype(str)
    by_month = (pred.groupby("month").agg(games=("correct", "size"), accuracy=("correct", "mean"))
                .reset_index())

    # Accuracy by team (games involving each team)
    long = pd.concat([
        pred.assign(team=pred.home_team, team_name=pred.home_team_name,
                    team_won=pred.home_win),
        pred.assign(team=pred.away_team, team_name=pred.away_team_name,
                    team_won=1 - pred.home_win)])
    by_team = (long.groupby(["team", "team_name"])
               .agg(games=("correct", "size"), accuracy=("correct", "mean"),
                    wins=("team_won", "sum")).reset_index()
               .sort_values("accuracy"))

    # Biggest upsets: most confident wrong calls
    upsets = pred[pred.correct == 0].sort_values("confidence", ascending=False).head(10)

    # ---------- export everything the dashboard needs ----------
    def rec(df):
        return json.loads(df.to_json(orient="records", date_format="iso"))

    tuning_best = tuning.sort_values("valid_log_loss").groupby("model").head(1)
    dash = {
        "meta": {
            "n_games_total": int(len(games)),
            "n_train": int(len(train_full)),
            "n_test": int(len(test)),
            "train_seasons": f"{seasons[1]} to {seasons[-2]}",
            "burn_in": BURN_IN,
            "test_season": TEST_SEASON,
            "valid_season": VALID_SEASON,
            "n_train_seasons": int(train_full.season.nunique()),
            "n_walk_seasons": int(walk.season.nunique()),
            "first_season": seasons[0],
            "last_season": seasons[-1],
            "best_model": best_name,
            "home_win_rate_train": float(train_full.home_win.mean()),
            "home_win_rate_test": float(test.home_win.mean()),
            "lr_params": lr_params,
            "gb_params": gb_params,
            "n_features": len(FEATURES),
        },
        "results": rec(results),
        "walk_forward": rec(walk[["season", "model", "accuracy", "log_loss"]]),
        "importance": rec(importance),
        "calibration": rec(calib),
        "by_conf": rec(by_conf.astype({"conf_bin": str})),
        "by_month": rec(by_month),
        "by_team": rec(by_team),
        "upsets": rec(upsets[["game_date", "home_team", "away_team", "home_pts", "away_pts",
                              "p_home", "confidence"]]),
        "games": rec(pred[["game_date", "home_team", "away_team", "home_pts", "away_pts",
                           "p_home", "correct"]].assign(p_home=lambda x: x.p_home.round(3))),
        "tuning_best": rec(tuning_best),
        "ablation": rec(ablation),
        "rest_split": rec(rest_split),
        "home_by_season": rec(games.groupby("season").home_win.mean().rename("home_win_rate")
                              .reset_index()),
    }
    with open(OUT / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(dash, f)
    print("\nWrote outputs to", OUT)


if __name__ == "__main__":
    main()
