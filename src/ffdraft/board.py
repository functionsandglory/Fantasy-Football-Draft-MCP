"""Draft board state and live sync with drafting platforms."""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from . import names, sources
from .config import CURRENT_SEASON, STATE_DIR, LeagueSettings

# Name handling lives in names.py so every join in the codebase resolves identically.
norm_name = names.normalize

_INDEX_CACHE: dict[str, names.PlayerIndex] = {}

# ESPN's proTeamId -> franchise name, from the proTeamSchedules view (stable
# reference data, not worth a network round trip on every sync). Team defenses
# in draft picks are encoded as -(16000 + proTeamId) rather than a real playerId.
_ESPN_PRO_TEAMS = {
    1: "Atlanta Falcons", 2: "Buffalo Bills", 3: "Chicago Bears",
    4: "Cincinnati Bengals", 5: "Cleveland Browns", 6: "Dallas Cowboys",
    7: "Denver Broncos", 8: "Detroit Lions", 9: "Green Bay Packers",
    10: "Tennessee Titans", 11: "Indianapolis Colts", 12: "Kansas City Chiefs",
    13: "Las Vegas Raiders", 14: "Los Angeles Rams", 15: "Miami Dolphins",
    16: "Minnesota Vikings", 17: "New England Patriots", 18: "New Orleans Saints",
    19: "New York Giants", 20: "New York Jets", 21: "Philadelphia Eagles",
    22: "Arizona Cardinals", 23: "Pittsburgh Steelers", 24: "Los Angeles Chargers",
    25: "San Francisco 49ers", 26: "Seattle Seahawks", 27: "Tampa Bay Buccaneers",
    28: "Washington Commanders", 29: "Carolina Panthers", 30: "Jacksonville Jaguars",
    33: "Baltimore Ravens", 34: "Houston Texans",
}


def _board_fingerprint(table: pd.DataFrame) -> str:
    """Cheap content signature. Keying on id() would be wrong as well as slow —
    CPython recycles ids, so a rebuilt board can land on a freed id and get served
    a stale index belonging to a different set of players."""
    if "name" not in table.columns or table.empty:
        return f"empty:{len(table)}"
    names_col = table["name"]
    return f"{len(table)}:{hash(names_col.iloc[0])}:{hash(names_col.iloc[-1])}"


def player_index(table: pd.DataFrame) -> names.PlayerIndex:
    """Alias index for a board, cached on the board's contents."""
    key = _board_fingerprint(table)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = names.PlayerIndex(table)
        _INDEX_CACHE.clear()   # only one board is live at a time
        _INDEX_CACHE[key] = idx
    return idx


def match_player(query: str, table: pd.DataFrame,
                 position: str | None = None) -> pd.Series | None:
    """Resolve a free-text name to a row, tolerating nicknames, suffixes and typos."""
    row, _ = player_index(table).resolve(query, position)
    return row


def match_player_verbose(query: str, table: pd.DataFrame,
                         position: str | None = None) -> tuple[pd.Series | None, str]:
    """Same, but also reports how the match was made."""
    return player_index(table).resolve(query, position)


# ---------------------------------------------------------------- ADP

FANTASYPROS_ADP = {
    "half_ppr": "https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php",
    "ppr": "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
    "standard": "https://www.fantasypros.com/nfl/adp/overall.php",
}


def load_adp(fmt: str = "half_ppr", csv_path: str | None = None,
             season: int = CURRENT_SEASON, superflex: bool = False) -> pd.DataFrame:
    """Draft-cost estimates, in order of preference.

    1. A CSV you export from your own platform — always best, because ADP is
       league- and format-specific and your room is what you're drafting against.
    2. FantasyPros preseason expert consensus rank, mirrored by dynastyprocess as a
       parquet going back to 2019. This is the reliable path: a direct data file
       rather than an HTML page that changes layout and blocks scripted requests.
    3. FantasyPros' live HTML page, as a last resort.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}
        name_c = next((cols[c] for c in ("name", "player", "player_name") if c in cols), None)
        adp_c = next((cols[c] for c in ("adp", "avg", "average", "rank") if c in cols), None)
        if not name_c or not adp_c:
            raise ValueError("ADP CSV needs a name column and an adp column")
        out = df[[name_c, adp_c]].rename(columns={name_c: "name", adp_c: "adp"})
        out["adp"] = pd.to_numeric(out["adp"], errors="coerce")
        out["_key"] = out["name"].map(norm_name)
        out["source"] = "csv"
        return out.dropna(subset=["adp"])

    try:
        from .adp import preseason_ecr
        ecr = preseason_ecr(season, superflex=superflex)
        if not ecr.empty:
            ecr = ecr[["name", "position", "adp", "_key"]].copy()
            ecr["source"] = "fantasypros_ecr_superflex" if superflex else "fantasypros_ecr"
            return ecr
    except Exception as exc:
        print(f"ECR history unavailable ({type(exc).__name__}); trying live page")

    url = FANTASYPROS_ADP.get(fmt, FANTASYPROS_ADP["half_ppr"])
    resp = requests.get(url, timeout=20, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    })
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    df = max(tables, key=len)
    cols = {str(c).lower(): c for c in df.columns}
    name_c = next((cols[c] for c in cols if "player" in c), df.columns[1])
    adp_c = next((cols[c] for c in cols if c in ("avg", "avg.", "adp")), df.columns[-1])
    out = df[[name_c, adp_c]].copy()
    out.columns = ["name", "adp"]
    # FantasyPros appends team and bye to the name cell.
    out["name"] = out["name"].astype(str).str.replace(r"\s*\([^)]*\)", "", regex=True)
    out["name"] = out["name"].str.replace(r"\s+[A-Z]{2,3}\s*\(?\d*\)?$", "", regex=True).str.strip()
    out["adp"] = pd.to_numeric(out["adp"], errors="coerce")
    out["_key"] = out["name"].map(norm_name)
    out["source"] = "fantasypros_html"
    return out.dropna(subset=["adp"])


# Typical 12-team draft position by positional rank, as adp = a * rank^b.
# Fitted to the shape of real half-PPR boards. This matters because draft rooms do
# not draft in value order: QBs and TEs slide well past their raw value, and using
# model rank as a stand-in for ADP would assume the room agrees with the model —
# which would make the whole opportunity-cost calculation circular.
# How much of the pure points-arithmetic shift to apply when converting consensus
# rankings between scoring formats. Below 1.0 because draft rooms price things the
# format doesn't change — consistency, positional scarcity, name recognition.
FORMAT_SHIFT_DAMPING = 0.6

SYNTHETIC_ADP_CURVE = {
    "RB": (2.00, 1.12),
    "WR": (2.80, 1.03),
    "TE": (18.0, 0.80),
    "QB": (22.0, 0.68),
}


def synthetic_adp(position: str, pos_rank: float, seasons_stale: float = 0.0) -> float:
    """Draft-cost estimate for a player missing from real ADP, from the model's own
    positional rank -- with no real market behind it, this is only trustworthy for
    someone still actually in the league.

    seasons_stale is how far behind the board's freshest player this one's last
    active season is (0 for someone who played as recently as anyone else on the
    board). A big flat penalty per season, not a multiplier on the base estimate:
    real drafters don't discount a year-old star by some percentage, they stop
    trusting him at all, because "didn't play last year" could mean retired, hurt
    long-term, or out of the league, and the box scores alone can't tell which.
    Without this, a retired player's still-strong last-known form could earn him
    the single best synthetic ADP on the board -- his own rank produces its own
    inflated market price -- which is what let a running back retired for two
    seasons come back as the model's runaway top recommendation in a backtest.
    """
    a, b = SYNTHETIC_ADP_CURVE.get(position, (3.0, 1.05))
    base = a * max(1.0, pos_rank) ** b
    return float(base + 200.0 * max(0.0, seasons_stale))


def attach_adp(board: pd.DataFrame, adp: pd.DataFrame | None) -> pd.DataFrame:
    """Join ADP onto the board, falling back to positional draft curves where missing."""
    b = board.copy()
    b["_key"] = b["name"].map(norm_name)
    if adp is not None and not adp.empty:
        b = b.merge(adp[["_key", "adp"]].drop_duplicates("_key"), on="_key", how="left")
        b["adp_source"] = np.where(b["adp"].notna(), "consensus", "modelled")
    else:
        b["adp"] = np.nan
        b["adp_source"] = "modelled"

    if "last_season" in b.columns:
        freshest = b["last_season"].max()
        stale = (freshest - b["last_season"]).clip(lower=0).fillna(0)
    else:
        stale = pd.Series(0.0, index=b.index)
    # A player off every team's depth chart has no real path to touches even
    # though he may have played as recently as anyone else on the board (so
    # last_season alone reads him as fresh) -- treat it as one stale season's
    # worth of synthetic-ADP burial so a fallback estimate never hands him a
    # top-of-board fake market price. Real ADP (the branch above) already
    # reflects this correctly when it exists; this only guards the fallback.
    if "off_roster" in b.columns:
        stale = stale + b["off_roster"].fillna(False).astype(bool).astype(float)
    fallback = [synthetic_adp(p, r, s)
               for p, r, s in zip(b["position"], b["pos_rank"], stale)]
    b["adp"] = b["adp"].fillna(pd.Series(fallback, index=b.index))
    b["adp_delta"] = b["adp"] - b["overall_rank"]
    return b


def convert_adp_format(board: pd.DataFrame, scoring_label: str) -> pd.DataFrame:
    """Shift PPR consensus rankings into this league's scoring format.

    Published consensus is full PPR — that is the only overall redraft ranking
    FantasyPros publishes. Feeding it straight into a half-PPR or standard league
    misprices exactly the players the format is about: a back who catches 60 passes
    is worth far less without a full point per reception, while a touchdown-dependent
    early-down back becomes relatively more valuable.

    The market ranking stays the anchor, because it encodes talent, situation and
    injury news no model captures. Only the format delta is applied, and that delta
    is arithmetic rather than opinion: half PPR is PPR minus half a point per catch,
    and each player's reception volume comes from his own projection. The adjustment
    is expressed as a shift in rank positions so it composes cleanly with ADP.
    """
    b = board.copy()
    b["adp_format"] = scoring_label
    if scoring_label == "ppr" or "proj_points_ppr" not in b.columns:
        return b

    rank_ppr = b["proj_points_ppr"].rank(ascending=False, method="min")
    rank_fmt = b["proj_points"].rank(ascending=False, method="min")
    # Positive shift = this format devalues him relative to PPR, so he goes later.
    #
    # Damped rather than applied whole. The raw rank delta is what pure points
    # arithmetic implies, but real draft rooms move less than that: they also price
    # consistency, positional scarcity and name recognition, none of which change
    # with the scoring format. Undamped, Derrick Henry went from ADP 38 to 1.0 in
    # standard — right direction, absurd magnitude.
    b["adp_shift"] = (rank_fmt - rank_ppr).fillna(0.0) * FORMAT_SHIFT_DAMPING
    b["adp_ppr"] = b["adp"]
    b["adp"] = (b["adp"] + b["adp_shift"]).clip(lower=1.0)
    b["adp_delta"] = b["adp"] - b["overall_rank"]
    return b


# ---------------------------------------------------------------- draft state

class DraftState:
    """Who's been taken, by whom, and whose turn it is.

    State is stored per league, so two drafts running in different leagues never
    read each other's picks.
    """

    def __init__(self, league: LeagueSettings, name: str | None = None):
        self.league = league
        key = re.sub(r"[^A-Za-z0-9_-]", "_", name or league.name or "default")
        self.path = STATE_DIR / f"draft_{key}.json"
        self.picks: list[dict] = []
        # Your slot comes from the league config, always. Reading it back from the
        # saved draft file meant reconfiguring to pick 11 and still being advised
        # for pick 6, because the stale value won.
        self.my_slot = league.draft_slot
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.picks = raw.get("picks", [])
            # Picks recorded under a different league size describe a different
            # draft entirely; discard rather than misinterpret them.
            if raw.get("teams") not in (None, league.teams):
                self.picks = []

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "picks": self.picks, "my_slot": self.my_slot,
            "teams": self.league.teams, "league": self.league.name,
            "updated": time.time(),
        }, indent=2))

    # -- mutation
    def record(self, player_name: str, overall: int | None = None,
               team_slot: int | None = None, player_id: str | None = None) -> dict:
        overall = overall or (len(self.picks) + 1)
        slot = team_slot if team_slot is not None else self.slot_for_pick(overall)
        pick = {"overall": overall, "slot": slot, "name": player_name, "player_id": player_id}
        self.picks = [p for p in self.picks if p["overall"] != overall] + [pick]
        self.picks.sort(key=lambda p: p["overall"])
        self.save()
        return pick

    def undo(self) -> dict | None:
        if not self.picks:
            return None
        p = self.picks.pop()
        self.save()
        return p

    def reset(self) -> None:
        self.picks = []
        self.save()

    # -- queries
    def slot_for_pick(self, overall: int) -> int:
        t = self.league.teams
        rnd = (overall - 1) // t + 1
        idx = (overall - 1) % t + 1
        return (t - idx + 1) if (self.league.snake and rnd % 2 == 0) else idx

    @property
    def on_the_clock(self) -> int:
        return len(self.picks) + 1

    def my_picks(self) -> list[int]:
        return self.league.picks_for_slot(self.my_slot)

    def next_pick_for_me(self, after: int | None = None) -> int | None:
        after = after or self.on_the_clock
        upcoming = [p for p in self.my_picks() if p >= after]
        return upcoming[0] if upcoming else None

    def pick_after_next(self) -> int | None:
        nxt = self.next_pick_for_me()
        if nxt is None:
            return None
        later = [p for p in self.my_picks() if p > nxt]
        return later[0] if later else None

    def taken_keys(self) -> set[str]:
        return {norm_name(p["name"]) for p in self.picks}

    def my_roster(self, board: pd.DataFrame) -> dict[str, int]:
        mine = [p for p in self.picks if p["slot"] == self.my_slot]
        counts: dict[str, int] = {}
        idx = board.set_index("_key")["position"].to_dict() if "_key" in board.columns else {}
        for p in mine:
            pos = idx.get(norm_name(p["name"]))
            if pos:
                counts[pos] = counts.get(pos, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "picks_made": len(self.picks),
            "on_the_clock": self.on_the_clock,
            "round": (self.on_the_clock - 1) // self.league.teams + 1,
            "my_slot": self.my_slot,
            "my_next_pick": self.next_pick_for_me(),
            "picks_until_my_turn": max(0, (self.next_pick_for_me() or 0) - self.on_the_clock),
        }


# ---------------------------------------------------------------- platform sync

def _id_crosswalk() -> pd.DataFrame:
    """gsis_id <-> espn_id / sleeper_id, from nflverse rosters.

    weekly_rosters has one row per player per week, and espn_id/sleeper_id are only
    reliably populated in some of those snapshots -- roughly a third of rows have a
    null espn_id even for players whose ID is known in other rows. Taking the first
    row per gsis_id (the old drop_duplicates) kept whichever snapshot happened to
    come first, which silently dropped the real ID for about a quarter of players --
    Bijan Robinson, Jahmyr Gibbs and De'Von Achane among them, verified against a
    2025 ESPN draft where they came back as unmatched ESPN#<id> picks. Grouping and
    taking the first non-null value per column, independently, uses whichever
    snapshot actually has the ID instead of gambling on row order.
    """
    r = sources.weekly_rosters()
    keep = [c for c in ("gsis_id", "espn_id", "sleeper_id", "full_name", "position") if c in r.columns]
    x = r[keep].dropna(subset=["gsis_id"])
    x = x.groupby("gsis_id", as_index=False).agg(
        lambda s: next((v for v in s if pd.notna(v)), np.nan))
    for c in ("espn_id", "sleeper_id"):
        if c in x.columns:
            x[c] = x[c].astype("string").str.replace(r"\.0$", "", regex=True)
    return x


def sync_sleeper(draft_id: str) -> list[dict]:
    """Pull picks from a Sleeper draft. Sleeper's draft API is public — no auth needed."""
    url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    xwalk = _id_crosswalk().set_index("sleeper_id")["full_name"].to_dict()
    out = []
    for p in resp.json():
        meta = p.get("metadata") or {}
        name = " ".join(filter(None, [meta.get("first_name"), meta.get("last_name")])).strip()
        name = name or xwalk.get(str(p.get("player_id")), "")
        out.append({
            "overall": p.get("pick_no"),
            "slot": p.get("draft_slot"),
            "name": name,
            "player_id": None,
            "position": meta.get("position"),
        })
    return sorted([o for o in out if o["name"]], key=lambda o: o["overall"] or 0)


def _espn_cookies(swid: str | None = None, espn_s2: str | None = None) -> dict:
    swid = swid or os.environ.get("ESPN_SWID")
    espn_s2 = espn_s2 or os.environ.get("ESPN_S2")
    if not (swid and espn_s2):
        return {}
    return {"SWID": swid if swid.startswith("{") else f"{{{swid}}}",
            "espn_s2": espn_s2}


def _espn_request_headers() -> dict:
    """Headers ESPN's own web client sends; some live-draft reads need them."""
    return {
        "User-Agent": "ffdraft-mcp/1.0",
        "Accept": "application/json, text/plain, */*",
        "X-Fantasy-Platform": "kona-PROD",
        "X-Fantasy-Source": "kona",
    }


def _espn_filled_pick_count(data: dict) -> int:
    raw = (data.get("draftDetail") or {}).get("picks") or []
    return sum(1 for p in raw if not _is_unfilled_espn_pick(p.get("playerId")))


def _fetch_espn_league(league_id: str, season: int,
                       swid: str | None = None, espn_s2: str | None = None) -> dict:
    """Fetch league JSON with draft, team, roster, and player views.

    Tries lm-api-reads first, then fantasy.espn.com. During a live draft the two
    hosts can disagree on how many slots have a real playerId — keep whichever
    reports more filled picks (with cookies when available).
    """
    cookies = _espn_cookies(swid, espn_s2)
    views = ["mDraftDetail", "mTeam", "mSettings", "mRoster", "kona_player_info"]
    headers = _espn_request_headers()
    paths = [
        (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
         f"/segments/0/leagues/{league_id}", {}),
        (f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
         f"/segments/0/leagues/{league_id}", {}),
    ]
    best: dict = {}
    best_filled = -1
    for base, extra in paths:
        resp = requests.get(base, params={"view": views, **extra},
                            cookies=cookies, timeout=20, headers=headers)
        if resp.status_code != 200:
            continue
        data = resp.json()
        filled = _espn_filled_pick_count(data)
        if filled > best_filled:
            best, best_filled = data, filled
    if best:
        return best
    # Pre-2018 seasons live on leagueHistory instead of the seasons path.
    if season >= 2018:
        resp = requests.get(paths[0][0], params={"view": views},
                            cookies=cookies, timeout=20, headers=headers)
        resp.raise_for_status()
        return resp.json()
    hist_url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/leagueHistory/"
                f"{league_id}")
    resp = requests.get(hist_url, params={"view": views, "seasonId": season},
                        cookies=cookies, timeout=20, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        if not data:
            return {}
        data = data[0]
    return data


def _is_unfilled_espn_pick(player_id) -> bool:
    """True when ESPN marks a draft slot as not yet picked.

    Unfilled slots use playerId -1 (or 0/None). Team defenses are also negative
    (e.g. -16033) and must not be treated as empty.
    """
    if player_id is None:
        return True
    try:
        pid = int(player_id)
    except (TypeError, ValueError):
        return True
    return pid == -1 or pid == 0


def _espn_player_name_map(data: dict) -> dict[str, str]:
    """playerId -> fullName from rosters and kona_player_info in the same payload."""
    out: dict[str, str] = {}
    for team in data.get("teams") or []:
        for entry in (team.get("roster") or {}).get("entries") or []:
            player = (entry.get("playerPoolEntry") or {}).get("player") or {}
            pid = player.get("id")
            name = player.get("fullName")
            if pid is not None and name:
                out[str(pid)] = name
    for item in data.get("players") or []:
        player = item.get("player") if isinstance(item, dict) else None
        player = player or item
        if not isinstance(player, dict):
            continue
        pid = player.get("id")
        name = player.get("fullName")
        if pid is not None and name:
            out[str(pid)] = name
    return out


def _espn_dst_name(player_id: int) -> str:
    """Resolve team defense picks. ESPN uses negative ids with different offsets."""
    for offset in (16000, 15000):
        team_id = -player_id - offset
        if team_id in _ESPN_PRO_TEAMS:
            return f"{_ESPN_PRO_TEAMS[team_id]} D/ST"
    return f"ESPN#{player_id} D/ST"


def _espn_team_slots(raw_picks: list[dict]) -> dict[int, int]:
    """Map ESPN teamId -> first-round draft slot (roundPickNumber)."""
    slots: dict[int, int] = {}
    for p in raw_picks:
        if p.get("roundId") != 1:
            continue
        if _is_unfilled_espn_pick(p.get("playerId")):
            continue
        team_id = p.get("teamId")
        slot = p.get("roundPickNumber")
        if team_id is not None and slot is not None:
            slots[int(team_id)] = int(slot)
    return slots


def _espn_overall_pick(p: dict, n_teams: int, fallback: int) -> int | None:
    overall = p.get("overallPickNumber")
    if overall:
        return int(overall)
    rnd = p.get("roundId")
    pick_in_round = p.get("roundPickNumber")
    if rnd is not None and pick_in_round is not None and n_teams:
        return (int(rnd) - 1) * n_teams + int(pick_in_round)
    return fallback


def _parse_espn_picks(data: dict, espn_map: dict[str, str],
                      roster_names: dict[str, str]) -> list[dict]:
    """Turn draftDetail.picks into normalized pick dicts."""
    raw = (data.get("draftDetail") or {}).get("picks") or []
    n_teams = len(data.get("teams") or []) or 1
    team_slots = _espn_team_slots(raw)
    sorted_raw = sorted(raw, key=lambda p: (
        p.get("roundId") or 0, p.get("roundPickNumber") or 0, p.get("overallPickNumber") or 0))
    out = []
    for i, p in enumerate(sorted_raw):
        pid = p.get("playerId")
        if _is_unfilled_espn_pick(pid):
            continue
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        # Team defenses aren't players -- no gsis_id, so they're never in the
        # crosswalk. ESPN encodes them as -(16000 + proTeamId) instead.
        if pid_int < 0:
            name = _espn_dst_name(pid_int)
        else:
            pid_key = str(pid_int)
            name = (espn_map.get(pid_key) or roster_names.get(pid_key) or f"ESPN#{pid_int}")
        overall = _espn_overall_pick(p, n_teams, i + 1)
        if not overall:
            continue
        team_id = p.get("teamId")
        slot = team_slots.get(int(team_id)) if team_id is not None else None
        out.append({
            "overall": overall,
            "slot": slot,
            "name": name,
            "player_id": None,
        })
    return sorted(out, key=lambda o: o["overall"])


def espn_draft_diagnostics(data: dict) -> dict:
    """Explain why sync_espn might return zero picks despite a draftDetail payload."""
    detail = data.get("draftDetail") or {}
    raw = detail.get("picks") or []
    filled = sum(1 for p in raw if not _is_unfilled_espn_pick(p.get("playerId")))
    drafted = detail.get("drafted")
    in_progress = detail.get("inProgress")
    hint = None
    if raw and not filled:
        if in_progress:
            hint = (
                "ESPN returned the full draft order (raw_slots) but every slot still has "
                "playerId -1 — the standard mDraftDetail endpoint usually does not stream "
                "live picks. The draft room uses a separate protocol; mDraftDetail typically "
                "fills in only after the draft ends (or lags badly mid-draft). For a live "
                "ESPN draft, use platform='paste' or record_pick between picks. Private "
                "leagues also need ESPN_SWID and ESPN_S2 set before the server starts."
            )
        elif not drafted:
            hint = (
                "The draft is scheduled but ESPN reports drafted=false — every slot still "
                "has playerId -1/0. That is normal before the draft starts (ESPN pre-allocates "
                "the pick order). Pass the correct season= if you meant a prior year."
            )
        else:
            hint = (
                "draftDetail has slots but no filled playerIds — check season= and that "
                "ESPN_SWID/ESPN_S2 are set for private leagues."
            )
    elif in_progress and filled:
        hint = (
            f"Live draft: {filled} of {len(raw)} slots have a playerId so far. Re-sync "
            "between picks; if the count stops updating, ESPN has stopped pushing picks to "
            "mDraftDetail and you will need platform='paste'."
        )
    return {
        "drafted": drafted,
        "in_progress": in_progress,
        "raw_slots": len(raw),
        "filled_slots": filled,
        "on_the_clock": filled + 1 if raw else None,
        "hint": hint,
    }


def sync_espn(league_id: str, season: int = CURRENT_SEASON,
              swid: str | None = None, espn_s2: str | None = None) -> list[dict]:
    """Pull picks from an ESPN league's draft detail endpoint.

    Public leagues work with no credentials. Private leagues need the SWID and
    espn_s2 cookies from a logged-in browser session, passed here or set as the
    ESPN_SWID / ESPN_S2 environment variables.
    """
    data = _fetch_espn_league(league_id, season, swid=swid, espn_s2=espn_s2)
    xwalk = _id_crosswalk()
    espn_map = xwalk.dropna(subset=["espn_id"]).set_index("espn_id")["full_name"].to_dict()
    return _parse_espn_picks(data, espn_map, _espn_player_name_map(data))


# ESPN's lineupSlotCounts slot ids that count as a flex, and which positions each
# is eligible for. Used to translate a real ESPN roster into LeagueSettings.starters.
_ESPN_FLEX_SLOTS = {"3": ("RB", "WR"), "5": ("WR", "TE"), "23": ("RB", "WR", "TE"),
                   "7": ("QB", "RB", "WR", "TE")}
_ESPN_BASE_SLOTS = {"0": "QB", "2": "RB", "4": "WR", "6": "TE", "16": "DST", "17": "K"}


def espn_league_context(league_id: str, season: int = CURRENT_SEASON,
                        swid: str | None = None, espn_s2: str | None = None) -> dict:
    """Everything needed to configure a league and find yourself in it, read
    straight from ESPN: team count, scoring, roster starters, your draft slot.

    Used by draft_backtest so a season/league_id is enough to run -- no manual
    configure_league bookkeeping for a season you're not actively drafting.
    """
    swid = swid or os.environ.get("ESPN_SWID")
    espn_s2 = espn_s2 or os.environ.get("ESPN_S2")
    data = _fetch_espn_league(league_id, season, swid=swid, espn_s2=espn_s2)
    settings = data.get("settings") or {}
    teams = data.get("teams") or []

    rec_item = next((i for i in settings.get("scoringSettings", {}).get("scoringItems", [])
                     if i.get("statId") == 53), None)
    rec_pts = float(rec_item["points"]) if rec_item else 0.0
    scoring = "ppr" if rec_pts >= 0.9 else "half_ppr" if rec_pts >= 0.35 else "standard"

    slot_counts = settings.get("rosterSettings", {}).get("lineupSlotCounts", {}) or {}
    starters = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "K": 0, "DST": 0}
    for sid, count in slot_counts.items():
        if sid in _ESPN_BASE_SLOTS and count:
            starters[_ESPN_BASE_SLOTS[sid]] += count
        elif sid in _ESPN_FLEX_SLOTS and count:
            starters["FLEX"] += count  # sub-eligibility isn't tracked, same as configure_league
    roster_slots = sum(int(v) for v in slot_counts.values())

    my_team = None
    if swid:
        target = swid.strip("{}")
        my_team = next((t for t in teams if target in [o.strip("{}") for o in t.get("owners", [])]),
                       None)
    draft_slot = None
    if my_team is not None:
        picks = (data.get("draftDetail") or {}).get("picks") or []
        mine = sorted([p for p in picks
                       if p.get("teamId") == my_team["id"]
                       and not _is_unfilled_espn_pick(p.get("playerId"))],
                      key=lambda p: p.get("overallPickNumber", 0))
        if not mine:
            # Fall back to round-1 slot order even before any picks are recorded.
            mine = sorted([p for p in picks if p.get("teamId") == my_team["id"]],
                          key=lambda p: p.get("overallPickNumber", 0))
        if mine:
            draft_slot = mine[0].get("roundPickNumber")

    return {
        "league_name": settings.get("name"),
        "teams": len(teams),
        "scoring": scoring,
        "starters": starters,
        "rounds": max(1, roster_slots),
        "my_team_id": my_team["id"] if my_team is not None else None,
        "draft_slot": draft_slot,
    }


_POS = r"(?:QB|RB|WR|TE|K|D/?ST|DEF)"
_TEAM = r"[A-Z]{2,4}"
_NOT_TEAM = frozenset({"II", "III", "IV", "JR", "SR", "V"})

# Standalone "R1, P1 - Team Name" lines from draft-room UIs — metadata, not a pick.
_LABEL_LINE = re.compile(
    rf"^\s*(?:R?\d+[.,]\s*P?\d+|round\s*\d+[,\s]*pick\s*\d+)\s*[-–—:]\s*(.+)$",
    re.I,
)
_ROUND_PREFIX = re.compile(
    r"^\s*(?:"
    r"(?:R?\d+[.,)\s]*P?\d+|\d+\.\d+)"  # R1, P1 / 1.1 / 45.
    r"|round\s*\d+[,\s]*pick\s*\d+"
    r")\s*[-–—:]?\s*",
    re.I,
)
_NUMBERED_PICK = re.compile(r"^\s*(\d+)\.\s+")


def _split_board_lines(text: str) -> list[str]:
    """Split pasted text into logical lines without breaking 'Round 1, Pick 7'."""
    if "\n" in text:
        return text.splitlines()
    protected = re.sub(
        r"(round\s*\d+),\s*(pick\s*\d+)",
        lambda m: f"{m.group(1)}\x00{m.group(2)}",
        text,
        flags=re.I,
    )
    return [p.replace("\x00", ",").strip() for p in re.split(r"[,;]+", protected) if p.strip()]


def _has_position_tag(s: str) -> bool:
    return bool(re.search(rf"\b{_POS}\b", s, re.I))


def _is_label_line(s: str) -> bool:
    """Round/pick header with a fantasy-team name and no player position."""
    m = _LABEL_LINE.match(s.strip())
    if not m:
        return False
    return not _has_position_tag(m.group(1))


def _extract_position(s: str) -> str | None:
    m = re.search(rf"\b({_POS})\b", s, re.I)
    return m.group(1).upper().replace("DEF", "DST") if m else None


def _strip_team_and_position(s: str) -> str:
    """Remove trailing TEAM/POS tags common in pasted draft boards."""
    s = re.sub(rf"\s*[/\-–—]\s*{_TEAM}\s+{_POS}\s*$", "", s, flags=re.I)
    s = re.sub(rf"\s+{_TEAM}\s+{_POS}\s*$", "", s, flags=re.I)
    s = re.sub(rf"\s*[-–—(]\s*{_TEAM}\)?\s+{_POS}\b.*$", "", s, flags=re.I)
    s = re.sub(rf"\s*[-–—(]\s*{_POS}\b.*$", "", s, flags=re.I)
    m = re.search(rf"\s+({_TEAM})\s*$", s)
    if m and m.group(1) not in _NOT_TEAM:
        s = s[:m.start()]
    return s.strip()


def _looks_like_player_name(s: str) -> bool:
    return len(s) > 2 and bool(re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", s))


def _parse_pick_line(s: str) -> dict | None:
    """Extract player name (and optional position) from one pasted line."""
    s = s.strip()
    if not s or _is_label_line(s):
        return None
    s = _NUMBERED_PICK.sub("", s)
    s = _ROUND_PREFIX.sub("", s)
    position = _extract_position(s)
    name = _strip_team_and_position(s)
    if not _looks_like_player_name(name):
        return None
    return {"name": name, "position": position}


def parse_pasted_board(text: str) -> list[dict]:
    """Best-effort parse of a pasted list of drafted players.

    Returns one dict per genuine pick line, each with ``overall`` (1-based line
    position among player lines only), ``name``, and optional ``position``.
    Label-only lines like ``R1, P1 - For the Sorg`` are skipped so they do not
    inflate the pick count or shift downstream overall_pick numbers.

    Handles numbered lists, inline and standalone ``Round 3, Pick 7 - Name``,
    comma-separated runs, and ``Name TEAM POS`` trailing tags.
    """
    entries: list[dict] = []
    for line in _split_board_lines(text):
        parsed = _parse_pick_line(line)
        if parsed is None:
            continue
        entries.append({"overall": len(entries) + 1, **parsed})
    return entries
