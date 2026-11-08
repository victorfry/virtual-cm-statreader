"""Core Pipeline Service: Orchestrates video sampling, screen classification,
OCR extraction, and match aggregation into a single reusable interface.
"""
import os
import sys
from typing import Optional, Callable

from src.core.video_sampler import VideoSampler
from src.parser.roster_matcher import RosterMatcher
from src.parser.match_aggregator import MatchAggregator
from src.models.match_data import MatchRecord


def parse_match_video(
    video_path: str,
    sample_fps: float = 10.0,
    roster_matcher: Optional[RosterMatcher] = None,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None
) -> MatchRecord:
    """Processes a match recording video from start to finish and returns a validated MatchRecord.
    
    Args:
        video_path: Path to the .mp4 or .mov video file.
        sample_fps: Sampling rate for stability detection (default: 10.0).
        roster_matcher: Optional RosterMatcher instance loaded with official squad rosters.
        progress_callback: Optional callback(current_idx, total_frames, timestamp_sec, status_text).

    Returns:
        MatchRecord with validated teams, scores, players, clean sheets, and goalkeeper stats.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Sample stable frames using UI anchor stability detection
    sampler = VideoSampler(sample_fps=sample_fps)
    stable_frames = sampler.sample_stable_frames(video_path)

    if not stable_frames:
        raise ValueError(f"No stable post-match frames detected in {video_path}. Ensure the video shows post-match stats menus.")

    # 2. Aggregate stable frames into a complete MatchRecord
    aggregator = MatchAggregator(roster_matcher=roster_matcher)
    match_record = aggregator.process_frames(stable_frames, progress_callback=progress_callback)

    return match_record
