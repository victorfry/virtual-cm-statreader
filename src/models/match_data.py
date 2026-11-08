"""Data models for FC match and player stats."""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class OutfieldPlayerStats(BaseModel):
    # Player identity
    full_name: str = Field(..., description="Full in-game player name")
    short_name: str = Field(..., description="Short name from squad list")
    team_name: str = Field(..., description="Team name")
    position: str = Field(..., description="Position played, e.g. ST, LW, CB")
    overall_rating: int = Field(..., description="Card overall rating (OVR)")
    match_rating: float = Field(..., description="Match rating, e.g. 6.5, 7.0")
    
    # Special toggles
    man_of_the_match: bool = Field(default=False, description="Whether player was awarded Man of the Match")
    clean_sheet: bool = Field(default=False, description="Whether team kept a clean sheet (0 goals conceded)")
    
    # 17 Outfield Match stats
    goals: int = Field(default=0, description="Goals scored")
    assists: int = Field(default=0, description="Assists")
    shots: int = Field(default=0, description="Total shots")
    shot_accuracy: int = Field(default=0, description="Shot accuracy percentage (0-100)")
    passes: int = Field(default=0, description="Total passes")
    pass_accuracy: int = Field(default=0, description="Pass accuracy percentage (0-100)")
    dribbles: int = Field(default=0, description="Total dribbles")
    dribble_success_rate: int = Field(default=0, description="Dribble success percentage (0-100)")
    tackles: int = Field(default=0, description="Total tackles")
    tackle_success_rate: int = Field(default=0, description="Tackle success percentage (0-100)")
    offsides: int = Field(default=0, description="Offsides")
    fouls_committed: int = Field(default=0, description="Fouls committed")
    possession_won: int = Field(default=0, description="Possession won")
    possession_lost: int = Field(default=0, description="Possession lost")
    minutes_played: int = Field(default=0, description="Minutes played")
    distance_covered_km: Optional[float] = Field(default=None, description="Distance covered in km")
    distance_sprinted_km: Optional[float] = Field(default=None, description="Distance sprinted in km")
    
    # Discipline
    yellow_cards: int = Field(default=0, description="Yellow cards received (0-2)")
    red_cards: int = Field(default=0, description="Red cards received (0-1)")


class GoalkeeperStats(BaseModel):
    """Goalkeeper-specific saving stats extracted from the Goalkeeping (Overall Saving) tab."""
    full_name: str = Field(..., description="Full in-game goalkeeper name")
    shots_against: int = Field(default=0, description="Total shots faced")
    shots_on_target: int = Field(default=0, description="Shots on target faced")
    saves: int = Field(default=0, description="Saves made")
    goals_conceded: int = Field(default=0, description="Goals conceded")
    save_success_rate: int = Field(default=0, description="Save success percentage (0-100)")
    
    # Additional detailed goalkeeping stats if present
    punch_saves: int = Field(default=0)
    rush_saves: int = Field(default=0)
    penalty_saves: int = Field(default=0)
    penalty_goals_conceded: int = Field(default=0)
    shootout_saves: int = Field(default=0)
    shootout_goals_conceded: int = Field(default=0)
    punch_clearances: int = Field(default=0)
    cross_claims: int = Field(default=0)
    calculated_gk_stats: Optional[bool] = Field(
        default=None,
        serialization_alias="calculatedGKStats",
        description="True if goalkeeping metrics were calculated due to missing GK tab"
    )


class TeamMatchStats(BaseModel):
    team_name: str
    score: int
    players: List[OutfieldPlayerStats] = Field(default_factory=list)
    goalkeeper: Optional[GoalkeeperStats] = Field(default=None)
    clean_sheet: bool = False
    
    # Team totals from Match Facts
    possession_pct: Optional[int] = None
    shots: Optional[int] = None
    expected_goals: Optional[float] = None
    passes: Optional[int] = None
    tackles: Optional[int] = None
    tackles_won: Optional[int] = None
    interceptions: Optional[int] = None
    saves: Optional[int] = None
    fouls: Optional[int] = None
    yellow_cards: Optional[int] = None


class MatchRecord(BaseModel):
    home_team: TeamMatchStats
    away_team: TeamMatchStats
    man_of_the_match_player: Optional[str] = None
    match_time: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
