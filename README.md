# NBA Game Outcome Predictor

Predicts the winner of NBA regular-season games using only information available **before tip-off**. It is trained on 14 seasons (2011-12 to 2024-25) and tested on the full **2025-26** season, the most recent complete season, which it never saw during training.

## Results: held-out 2025-26 season (1,230 games)

| Approach | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| **Logistic regression (final model)** | **68.9%** | **0.595** | **0.204** | **0.741** |
| Ensemble (LR + gradient boosting) | 69.0% | 0.596 | 0.204 | 0.741 |
| Gradient boosting | 68.5% | 0.597 | 0.205 | 0.740 |
| Elo ratings only | 67.6% | 0.607 | 0.209 | 0.738 |
| Better record wins | 67.3% | 0.633 | 0.221 | 0.675 |
| Always pick the home team | 55.4% | 0.688 | 0.247 | n/a |

Logistic regression (C = 0.03) is the final model because it had the best log loss on the 2024-25 validation season. On the test season the ensemble gets two more games right (849 vs 847), but its probabilities are no better, so the simpler model was kept.

- When the model is **80%+ confident**, it is right **85.1%** of the time (201 games).
- The model is well **calibrated**: when it gives a home team about 75%, the home team wins about 76% of the time.
- **Walk-forward validation** retrains the model for each season from 2013-14 to 2025-26 on all earlier seasons. The model averages **65.9%**, against 65.3% for Elo alone, and beats or ties Elo in 9 of 13 seasons. Over the last three seasons its edge over Elo was +1.9, +2.4 and +1.3 points.

## Key findings

1. **Team strength does almost all the work.** A single Elo rating gap already reaches 65.7% (walk-forward). The other 24 features add about 0.2 points of accuracy.
2. **Rest matters.** Home teams win 61.7% of games when only the visitor is on a back-to-back, and 50.1% when only the home team is. Rest and schedule features give the clearest improvement over Elo.
3. **Home-court advantage is shrinking.** Home teams won 60.4% of games in 2010-11 and about 54–55% in each of the last three seasons.
4. **Accuracy climbs through the season.** In 2025-26 the model was right 79% of the time in March and April, against 64% in the first half of the season.
5. **The ceiling is roughly two-thirds to 70%.** Injuries, rest days and lineup news don't appear in box scores. Most of the biggest misses were strong home teams losing to weak visitors.

## Data

| Seasons | Source | Notes |
|---|---|---|
| 2010-11 to 2023-24 | [NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024) by Vitalii Korolyk (MIT licence), sourced from NBA.com | Team game totals |
| 2024-25 and 2025-26 | ESPN team box scores from [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_team_boxscores) | Converted to the NBA.com layout by `add_espn_seasons.py` |

The two sources were checked against each other on the overlapping 2023-24 season. All 2,460 team-games line up, and points, made shots, free throws, results and home/away are identical in 100% of rows. Field-goal attempts, offensive rebounds and turnovers are identical in 99.8% or more. The ESPN conversion drops the All-Star Game and the NBA Cup final, since neither counts in the standings.

## Features (25, all computed from earlier games only)

| Group | Features |
|---|---|
| Elo | Rating gap (K=20, 100-point home advantage, margin-of-victory multiplier, 25% regression to the mean each summer) |
| Season strength | Win %, net / offensive / defensive rating, point margin |
| Four Factors & style | eFG%, turnover rate, offensive and defensive rebounding %, free-throw rate (for and against), pace, 3-point attempt rate |
| Recent form | Last-10 win %, net, offensive and defensive rating |
| Rest & schedule | Rest days, back-to-back flags, games in the previous 7 days |

Season-to-date stats are **shrunk toward last season's values** (regressed 30% toward the league average) using a 10-game prior. That way, early-season predictions don't overreact to 2–3 games.

## Avoiding data leakage

- Every rolling or season stat is computed with `shift(1)`, so a game never sees its own result.
- Elo ratings are recorded before each game and updated after it.
- Model settings were chosen on 2024-25 only. The 2025-26 test season was scored once, at the end.
- 2010-11 is used only as a burn-in season to warm up Elo and the prior-season values.

## Run it

```bash
pip install -r requirements.txt   # pandas, numpy, scikit-learn, pyarrow
python add_espn_seasons.py   # adds 2024-25 and 2025-26 -> data/regular_season_totals_2010_2026.csv
python nba_predictor.py      # builds features, trains and evaluates the models, writes ./outputs
python build_dashboard.py    # builds NBA_Game_Predictor.html from the outputs
```

The whole pipeline runs in under a minute on a laptop. The combined data file is already included, so `add_espn_seasons.py` only needs re-running to refresh it.

## Files

| File | What it is |
|---|---|
| `nba_predictor.py` | The full pipeline: load → features → Elo → models → evaluation |
| `add_espn_seasons.py` | Converts and validates the ESPN seasons, and writes the combined data file |
| `build_dashboard.py` / `dashboard_template.html` | Builds the interactive results dashboard |
| `NBA_Game_Predictor.html` | The dashboard (open in a browser) |
| `outputs/predictions_2025_26.csv` | Every test-season game with the model's probability and whether it was right |
| `outputs/walk_forward.csv` | Per-season results for every approach |
| `outputs/feature_group_ablation.csv` | What each feature group adds |
| `outputs/tuning_results.csv` | Validation scores for every hyperparameter setting |
| `data/regular_season_totals_2010_2026.csv` | Combined team box scores, 19,118 games |
| `data/regular_season_totals_2010_2024.csv`, `data/espn/` | The original source files |

## Ideas for next steps

- Use player box scores to build lineup and availability features, weighting each team by who actually played
- Benchmark against closing betting lines
- Predict point margin as well as the winner
- Add playoff and play-in games (the ESPN files already include them)
- Re-run in spring 2027 with a 2026-27 file to score the current season as it happens
