import argparse
import json
import os
import sys
import time
from typing import Optional

from src.core.pipeline import parse_match_video
from src.parser.roster_matcher import RosterMatcher


def print_match_summary(record):
    """Pretty-print extracted match data to console."""
    print("\n" + "=" * 80)
    print(f"MATCH: {record.home_team.team_name} {record.home_team.score} - {record.away_team.score} {record.away_team.team_name} ({record.match_time})")
    if record.man_of_the_match_player:
        print(f"MAN OF THE MATCH: {record.man_of_the_match_player}")
    print("=" * 80)

    for team in [record.home_team, record.away_team]:
        print(f"\n--- {team.team_name.upper()} ({len(team.players)} players) ---")
        header = f"{'Pos':<5} {'Name':<20} {'OVR':<4} {'Rate':<5} {'G':<3} {'A':<3} {'Shots':<8} {'Pass':<8} {'Drib':<8} {'Tckl':<8} {'PW':<4} {'PL':<4} {'Off':<4} {'Fls':<4} {'YC':<3} {'RC':<3} {'Min':<4} {'MOTM':<5}"
        print(header)
        print("-" * len(header))

        for p in team.players:
            motm_str = "YES" if p.man_of_the_match else "-"
            shot_str = f"{p.shots}({p.shot_accuracy}%)"
            pass_str = f"{p.passes}({p.pass_accuracy}%)"
            drib_str = f"{p.dribbles}({p.dribble_success_rate}%)"
            tckl_str = f"{p.tackles}({p.tackle_success_rate}%)"
            print(f"{p.position:<5} {p.full_name[:18]:<20} {p.overall_rating:<4} {p.match_rating:<5.1f} {p.goals:<3} {p.assists:<3} {shot_str:<8} {pass_str:<8} {drib_str:<8} {tckl_str:<8} {p.possession_won:<4} {p.possession_lost:<4} {p.offsides:<4} {p.fouls_committed:<4} {p.yellow_cards:<3} {p.red_cards:<3} {p.minutes_played:<4} {motm_str:<5}")

        if team.goalkeeper:
            gk = team.goalkeeper
            gk_tag = " (Calculated Stats - GK Tab Missing)" if gk.calculated_gk_stats else ""
            print(f"\n[Goalkeeper: {gk.full_name}{gk_tag}]")
            print(f"  Shots Faced: {gk.shots_against} | On Target: {gk.shots_on_target} | Saves: {gk.saves} | Conceded: {gk.goals_conceded} | Save%: {gk.save_success_rate}%")


def main():
    parser = argparse.ArgumentParser(description="Parse EA Sports FC match video for player and match stats.")
    parser.add_argument("--video", required=True, help="Path to video file (.mp4, .mov)")
    parser.add_argument("--output", default="match_stats.json", help="Path to save extracted JSON stats")
    parser.add_argument("--roster-html", help="Path to HTML file containing squad select options")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Sampling FPS for stability detection (default: 10.0)")

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Starting extraction for video: {args.video}", flush=True)
    start_time = time.time()

    def on_progress(idx: int, total: int, timestamp_sec: float, msg: str):
        print(f"  [{idx:2d}/{total:2d}] ({timestamp_sec:5.2f}s) {msg}", flush=True)

    match_record = parse_match_video(
        video_path=args.video,
        sample_fps=args.sample_fps,
        progress_callback=on_progress
    )

    print_match_summary(match_record)

    out_dict = match_record.model_dump(by_alias=True, exclude_none=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n[+] Extraction completed in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    main()
