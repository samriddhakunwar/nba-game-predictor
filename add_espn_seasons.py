"""
Extend the dataset with the 2024-25 and 2025-26 regular seasons.

The main dataset (NBA.com via NocturneBear/NBA-Data-2010-2024) stops after
2023-24. Newer seasons come from ESPN team box scores published by the
sportsdataverse project (github.com/sportsdataverse/sportsdataverse-data,
release "espn_nba_team_boxscores").

This script:
  1. converts ESPN regular-season box scores into the NBA.com column layout
  2. drops games that don't count in the standings (All-Star, NBA Cup final)
  3. checks the conversion against the overlapping 2023-24 season
  4. writes data/regular_season_totals_2010_2026.csv

Run:  python add_espn_seasons.py   (downloads ~0.4 MB if files are missing)
"""
from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ESPN_DIR = DATA / "espn"
BASE = DATA / "regular_season_totals_2010_2024.csv"
OUT = DATA / "regular_season_totals_2010_2026.csv"
URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "espn_nba_team_boxscores/team_box_{year}.parquet")

# ESPN season label = year the season ends (2025 -> 2024-25)
NEW_SEASONS = {2025: "2024-25", 2026: "2025-26"}
ESPN_TO_NBA = {"WSH": "WAS", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "GS": "GSW"}


def fetch(year: int) -> pd.DataFrame:
    ESPN_DIR.mkdir(parents=True, exist_ok=True)
    path = ESPN_DIR / f"team_box_{year}.parquet"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(year=year), tmp)
        tmp.replace(path)  # only a complete download gets the real name
    return pd.read_parquet(path)


def espn_regular_season(year: int, nba_teams: set[str]) -> pd.DataFrame:
    e = fetch(year)
    e = e[e.season_type == 2].copy()  # 2 = regular season (3 = playoffs, 5 = play-in)
    e["team"] = e.team_abbreviation.replace(ESPN_TO_NBA)
    e["opp"] = e.opponent_team_abbreviation.replace(ESPN_TO_NBA)
    # All-Star teams (EAST/WEST, and the 2025+ mini-tournament teams) are not NBA teams
    e = e[e.team.isin(nba_teams) & e.opp.isin(nba_teams)]
    # The NBA Cup final is labelled regular season by ESPN but doesn't count in the
    # standings: it is the only game on its date, between the two teams with 83 games.
    games_per_team = e.team.value_counts()
    finalists = games_per_team[games_per_team > 82].index
    games_per_day = e.groupby("game_date").game_id.nunique()
    cup_final = e[e.team.isin(finalists) & e.opp.isin(finalists)
                  & e.game_date.map(games_per_day).eq(1)].game_id.unique()
    e = e[~e.game_id.isin(cup_final)]
    assert e.team.value_counts().eq(82).all() and e.team.nunique() == 30, f"{year}: bad team counts"
    return e


def to_nba_layout(e: pd.DataFrame, season: str, team_ids: dict, team_names: dict) -> pd.DataFrame:
    home = e.team_home_away.eq("home")
    out = pd.DataFrame({
        "SEASON_YEAR": season,
        "TEAM_ID": e.team.map(team_ids),
        "TEAM_ABBREVIATION": e.team,
        "TEAM_NAME": e.team.map(team_names),
        "GAME_ID": e.game_id.astype(int),
        "GAME_DATE": pd.to_datetime(e.game_date).dt.strftime("%Y-%m-%dT00:00:00"),
        "MATCHUP": e.team + home.map({True: " vs. ", False: " @ "}) + e.opp,
        "WL": e.team_winner.map({True: "W", False: "L"}),
        "MIN": 48.0,  # ESPN has no team minutes; only used for pace (overtime is rare)
        "PTS": e.team_score,
        "FGM": e.field_goals_made, "FGA": e.field_goals_attempted,
        "FG3M": e.three_point_field_goals_made, "FG3A": e.three_point_field_goals_attempted,
        "FTM": e.free_throws_made, "FTA": e.free_throws_attempted,
        "OREB": e.offensive_rebounds, "DREB": e.defensive_rebounds,
        "AST": e.assists, "TOV": e.total_turnovers,  # total = player + team turnovers, as NBA.com
        "STL": e.steals, "BLK": e.blocks, "PF": e.fouls,
    })
    return out.reset_index(drop=True)


def validate_against_nba(base: pd.DataFrame, team_ids: dict, team_names: dict) -> None:
    """Convert ESPN 2023-24 and compare with the NBA.com rows for the same games."""
    nba_teams = set(team_ids)
    e = to_nba_layout(espn_regular_season(2024, nba_teams), "2023-24", team_ids, team_names)
    n = base[base.SEASON_YEAR == "2023-24"]
    m = n.merge(e, on=["GAME_DATE", "TEAM_ABBREVIATION"], suffixes=("_nba", "_espn"))
    assert len(m) == len(n) == len(e) == 2460, "2023-24 games don't line up"
    print("ESPN vs NBA.com, 2023-24 (2,460 team-games):")
    for c in ["PTS", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV", "WL", "MATCHUP"]:
        rate = (m[f"{c}_nba"] == m[f"{c}_espn"]).mean()
        print(f"  {c:<8} identical in {rate:.2%} of rows")
        assert rate > 0.99, c


def main() -> None:
    base = pd.read_csv(BASE)
    latest = base.sort_values("GAME_DATE").drop_duplicates("TEAM_ABBREVIATION", keep="last")
    team_ids = dict(zip(latest.TEAM_ABBREVIATION, latest.TEAM_ID))
    team_names = dict(zip(latest.TEAM_ABBREVIATION, latest.TEAM_NAME))

    validate_against_nba(base, team_ids, team_names)

    new = [to_nba_layout(espn_regular_season(y, set(team_ids)), s, team_ids, team_names)
           for y, s in NEW_SEASONS.items()]
    full = pd.concat([base[new[0].columns]] + new, ignore_index=True)
    assert full.groupby("GAME_ID").size().eq(2).all()
    full.to_csv(OUT, index=False, lineterminator="\n")
    print("\nGames per season:")
    print((full.groupby("SEASON_YEAR").size() // 2).to_string())
    print(f"\nWrote {OUT} ({len(full) // 2:,} games)")


if __name__ == "__main__":
    main()
