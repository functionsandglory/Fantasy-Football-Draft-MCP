# Troubleshooting

## Setup

**`-bash: pip: command not found`**
Your system only has `python3`/`pip3` on PATH, or pip isn't installed at all. Use
`python3 -m pip install -e .`, or better, set up a virtual environment (see below) so
plain `python`/`pip` work for the rest of these commands.

**`error: externally-managed-environment`**
macOS with Homebrew Python (and some Linux distros) block system-wide `pip install` by
design (PEP 668). Use a virtual environment instead of `--break-system-packages`:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```
Activate it (`source .venv/bin/activate`) in every new terminal before running
`pytest`, `ruff`, or the server. Point `claude_desktop_config.json` at
`.venv/bin/python`, not your system Python, or Claude Desktop won't see the install.

**`ModuleNotFoundError: No module named 'ffdraft'`**
Install in editable mode from the repo root: `pip install -e .`

**`ImportError: cannot import name 'FastMCP'`**
The MCP SDK renamed its server class between 1.x and 2.x. The server handles both, so
this usually means a partial install: `pip install --upgrade mcp`

**`pyarrow` errors on first run**
Parquet needs it: `pip install pyarrow`

**The server doesn't appear in Claude**
Check the config path is right for your OS, use an absolute path to the Python that has
`ffdraft` installed, and restart Claude fully. Verify the server runs standalone:
`python -m ffdraft.server` — it should start and wait on stdio.

**`mcp-server-fantasy-draft.log` shows `ModuleNotFoundError: No module named 'ffdraft'`
even though `pip install -e .` succeeded and the server runs fine from the terminal**
macOS + Python 3.13 specific. The editable install works by dropping a `.pth` file (e.g.
`__editable__.ffdraft_mcp-1.0.0.pth`) in the venv's `site-packages` that points at `src/`.
Python 3.13 added a safety check that silently skips `.pth` files with the OS "hidden" flag
set, and this file can end up hidden (`ls -lO` on it shows `hidden` in the flags column) —
run `python -v -c "import ffdraft"` and look for `Skipping hidden .pth file` to confirm.

The usual cause: the repo (and its `.venv`) sits under an iCloud-synced folder — Desktop or
Documents with macOS's "Desktop & Documents Folders" sync turned on. iCloud's file provider
daemon tags newly written files hidden while it syncs them, including the `.pth` file the
moment `pip install -e .` creates it, and it will keep re-hiding it, so `chflags nohidden
<file>` only fixes it until the next sync pass.

The durable fix: put the venv somewhere not synced by iCloud, e.g. `~/.venvs/ffdraft-mcp`
instead of `<repo>/.venv`, and point `claude_desktop_config.json`'s `command` at that
`bin/python`. If you'd rather keep the venv where it is, add
`"PYTHONPATH": "/absolute/path/to/ff-draft-mcp/src"` to the server's `env` block in
`claude_desktop_config.json` — this bypasses the `.pth` mechanism entirely, regardless of
the flag's state, though it only patches the MCP server launch, not `python setup_data.py`
or other commands run directly from an activated shell.

## Data

**First run takes minutes**
Expected. It downloads about 200 MB of play-by-play. Run `prewarm` before draft day so
this never happens on the clock.

**`no seasons loaded for .../player_stats_YYYY.parquet`**
nflverse renamed this release to `stats_player_week` from 2025. Both layouts are handled,
so this means the season isn't published yet. Set `FFDRAFT_SEASONS` to seasons that exist.

**Stale data mid-season**
`refresh_data(force_download=true)`

**`team_context`'s `drive_efficiency` or `redzone_identity` comes back empty**
Those need the `fixed_drive_result`/`drive` play-by-play columns, added to `PBP_COLS`
after some cached `play_by_play` parquets were already built. A cache built before that
change won't have them until it's rebuilt: `refresh_data(force_download=true)`.

**Reclaiming disk space**
Delete `~/.ffdraft/cache/`. It rebuilds on the next run.

## Leagues

**Recommendations are for the wrong pick**
Check `draft_status`. If `my_slot` is wrong, re-run `configure_league` with the right
`draft_slot` — league config is authoritative.

**One league shows another's picks**
Shouldn't happen — state is per league. Confirm with `list_leagues` that you're on the
league you think, and that `picks_recorded` looks right.

**Switching leagues rebuilds the board**
Only when the format differs. Leagues sharing scoring, size and roster settings share a
cached board; only draft state and your slot differ.

## Drafting

**A drafted player still shows as available**
The name didn't resolve. Check with `resolve_names`, then record using the resolved
spelling. Report the miss as an issue — the alias map grows from real cases.

**"ambiguous (2): Justin Jefferson, Van Jefferson"**
Working as intended. Two players match; give a first name.

**Sleeper sync returns nothing**
Verify the draft id from the URL (`sleeper.com/draft/nfl/<draft_id>`) — that's the draft
id, not the league id. Mock drafts have their own ids.

**ESPN sync returns 401 or empty**
Private leagues need `ESPN_SWID` and `ESPN_S2` set before the server starts. Cookies
expire — log into ESPN and copy fresh ones. See [SECURITY.md](../SECURITY.md).

**ESPN live draft shows `picks_synced: 0` but the API response has a big `picks` array**
Normal for a live or not-yet-started draft. ESPN pre-allocates every slot in
`draftDetail.picks` with `playerId: -1` before anyone picks; the array length is
teams × rounds, not picks made so far. The standard `mDraftDetail` endpoint usually
does not stream live picks — the draft room uses a separate protocol, and playerIds
often only fill in after the draft ends (or lag badly mid-draft). Check
`espn_draft.filled_slots` in the sync response: if it is 0 while you are mid-draft,
use `platform="paste"` or `record_pick` between picks. Sleeper's public draft API
is fully live if your league drafts there instead.

**ESPN picks show as `ESPN#12345`**
The player id didn't map. Usually a rookie missing from the crosswalk. Record manually.

## Results that look wrong

**A young player is flagged "overvalued"**
The model is backward-looking, so it fades rookies and second-year players. Check
`player_report` for games played before acting on it.

**Rookies rank lower than expected**
By design. Their consistency prior is deliberately low and projections cap at the
observed 75th percentile for their draft slot. See
[methodology](methodology.md#rookies).

**Quarterbacks rank higher than they'd go in a real draft**
Correct in raw value. `who_should_i_pick` applies opportunity cost and roster need on top,
which is why it won't recommend one early in a 1-QB league. In superflex it will.

**Someone missing from `separation_report`**
Under 250 estimated routes or 50 targets. Rate stats on part-time usage are misleading.

**Model disagrees with my rankings**
It's opinionated. Tune with `model_settings` — raise `consistency_weight` for floor,
`injury_weight` if injuries burned you.

## Getting help

Open an issue with what you ran, what you expected, what happened, plus your league
format and the output of `list_leagues`. **Never include ESPN cookies.**
