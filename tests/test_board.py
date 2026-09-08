"""ESPN id crosswalk: tested offline with synthetic weekly_rosters data."""
import pandas as pd

from ffdraft import board, sources



class TestParsePastedBoard:
    def test_skips_standalone_label_lines(self):
        text = """R1, P1 - For the Sorg
Ja'Marr Chase CIN WR
R1, P2 - Other Team
Christian McCaffrey SF RB"""
        entries = board.parse_pasted_board(text)
        assert len(entries) == 2
        assert entries[0] == {"overall": 1, "name": "Ja'Marr Chase", "position": "WR"}
        assert entries[1] == {"overall": 2, "name": "Christian McCaffrey", "position": "RB"}

    def test_strips_trailing_team_and_position(self):
        text = """Puka Nacua LAR WR
Josh Allen BUF QB
James Cook III BUF RB
Kyle Pitts Sr. ATL TE"""
        entries = board.parse_pasted_board(text)
        names = [e["name"] for e in entries]
        assert names == ["Puka Nacua", "Josh Allen", "James Cook III", "Kyle Pitts Sr."]

    def test_preserves_pick_position_when_names_fail_to_match(self):
        """Unmatched names must keep their line slot so downstream picks don't shift."""
        text = """Player One AAA WR
Player Two BBB WR
Player Three CCC WR"""
        entries = board.parse_pasted_board(text)
        assert [e["overall"] for e in entries] == [1, 2, 3]

    def test_inline_round_pick_with_player(self):
        text = "Round 3, Pick 7 - Ja'Marr Chase CIN WR"
        entries = board.parse_pasted_board(text)
        assert len(entries) == 1
        assert entries[0]["name"] == "Ja'Marr Chase"

    def test_comma_separated_without_newlines(self):
        text = "Ja'Marr Chase CIN WR, Christian McCaffrey SF RB"
        entries = board.parse_pasted_board(text)
        assert len(entries) == 2

    def test_round_comma_pick_not_split_on_comma(self):
        text = "Round 1, Pick 7 - Ja'Marr Chase CIN WR"
        entries = board.parse_pasted_board(text)
        assert len(entries) == 1
        assert entries[0]["overall"] == 1

    def test_numbered_list(self):
        text = """45. Bucky Irving TB RB
46. Cam Skattebo NYG RB"""
        entries = board.parse_pasted_board(text)
        assert len(entries) == 2
        assert entries[0]["name"] == "Bucky Irving"
        assert entries[1]["name"] == "Cam Skattebo"
        assert entries[0]["overall"] == 1
        assert entries[1]["overall"] == 2


def _sample_espn_payload(*pick_rows, teams=10):
    return {
        "teams": [{"id": i} for i in range(1, teams + 1)],
        "draftDetail": {
            "drafted": True,
            "inProgress": False,
            "picks": list(pick_rows),
        },
    }


class TestEspnDraftParsing:
    def test_skips_unfilled_slots(self):
        payload = _sample_espn_payload(
            {"playerId": -1, "overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
             "teamId": 1},
            {"playerId": 4430807, "overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2,
             "teamId": 2},
        )
        picks = board._parse_espn_picks(payload, {}, {})
        assert len(picks) == 1
        assert picks[0]["overall"] == 2

    def test_skips_player_id_zero(self):
        payload = _sample_espn_payload(
            {"playerId": 0, "overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
             "teamId": 1},
            {"playerId": 123, "overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2,
             "teamId": 2},
        )
        picks = board._parse_espn_picks(payload, {"123": "Test Player"}, {})
        assert [p["overall"] for p in picks] == [2]

    def test_derives_overall_when_missing(self):
        payload = _sample_espn_payload(
            {"playerId": 100, "roundId": 1, "roundPickNumber": 1, "teamId": 1},
            {"playerId": 200, "roundId": 1, "roundPickNumber": 2, "teamId": 2},
        )
        picks = board._parse_espn_picks(payload, {"100": "A", "200": "B"}, {})
        assert [p["overall"] for p in picks] == [1, 2]

    def test_resolves_names_from_roster_in_payload(self):
        payload = _sample_espn_payload(
            {"playerId": 999, "overallPickNumber": 1, "roundId": 1, "roundPickNumber": 3,
             "teamId": 3},
        )
        payload["teams"] = [{
            "id": 3,
            "roster": {"entries": [{"playerPoolEntry": {"player": {"id": 999,
                                                                   "fullName": "Roster Guy"}}}]},
        }]
        picks = board._parse_espn_picks(payload, {}, board._espn_player_name_map(payload))
        assert picks[0]["name"] == "Roster Guy"
        assert picks[0]["slot"] == 3

    def test_diagnostics_for_live_draft_with_only_unfilled_slots(self):
        payload = {
            "draftDetail": {
                "drafted": False,
                "inProgress": True,
                "picks": [{"playerId": -1, "overallPickNumber": i, "teamId": 1}
                          for i in range(1, 4)],
            },
        }
        diag = board.espn_draft_diagnostics(payload)
        assert diag["raw_slots"] == 3
        assert diag["filled_slots"] == 0
        assert "live draft" in (diag["hint"] or "").lower() or "mDraftDetail" in (diag["hint"] or "")

    def test_dst_negative_id_not_treated_as_unfilled(self):
        payload = _sample_espn_payload(
            {"playerId": -15012, "overallPickNumber": 1, "roundId": 1,
             "roundPickNumber": 1, "teamId": 1},
        )
        picks = board._parse_espn_picks(payload, {}, {})
        assert len(picks) == 1
        assert picks[0]["name"] == "Kansas City Chiefs D/ST"


class TestIdCrosswalk:
    def test_prefers_row_with_espn_id_over_earlier_null_row(self, monkeypatch):
        # weekly_rosters has one row per player per week; espn_id/sleeper_id are
        # only populated in some of those snapshots. A player whose earliest row
        # happens to lack espn_id must still resolve to the ID a later row has --
        # this is what silently dropped Bijan Robinson, Jahmyr Gibbs and De'Von
        # Achane (~23% of a real draft) before the fix.
        rosters = pd.DataFrame([
            {"gsis_id": "00-0038542", "espn_id": None, "sleeper_id": "9999",
             "full_name": "Bijan Robinson", "position": "RB"},
            {"gsis_id": "00-0038542", "espn_id": "4430807", "sleeper_id": None,
             "full_name": "Bijan Robinson", "position": "RB"},
        ])
        monkeypatch.setattr(sources, "weekly_rosters", lambda: rosters)

        x = board._id_crosswalk().set_index("gsis_id")
        assert x.loc["00-0038542", "espn_id"] == "4430807"
        assert x.loc["00-0038542", "sleeper_id"] == "9999"

    def test_one_row_per_gsis_id(self, monkeypatch):
        rosters = pd.DataFrame([
            {"gsis_id": "00-0038542", "espn_id": None, "sleeper_id": None,
             "full_name": "Bijan Robinson", "position": "RB"},
            {"gsis_id": "00-0038542", "espn_id": "4430807", "sleeper_id": None,
             "full_name": "Bijan Robinson", "position": "RB"},
            {"gsis_id": "00-0038542", "espn_id": None, "sleeper_id": "9999",
             "full_name": "Bijan Robinson", "position": "RB"},
        ])
        monkeypatch.setattr(sources, "weekly_rosters", lambda: rosters)

        x = board._id_crosswalk()
        assert len(x) == 1

    def test_drops_players_with_no_gsis_id(self, monkeypatch):
        rosters = pd.DataFrame([
            {"gsis_id": None, "espn_id": "123", "sleeper_id": None,
             "full_name": "No Gsis Guy", "position": "WR"},
        ])
        monkeypatch.setattr(sources, "weekly_rosters", lambda: rosters)

        assert board._id_crosswalk().empty
