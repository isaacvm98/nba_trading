"""
Injury Data Provider using RapidAPI Tank01

Fetches current injury status for NBA teams and calculates impact scores
based on player importance (average minutes played).
"""

import requests
import time
from datetime import datetime, timezone


class InjuryProvider:
    """Fetches injury data from RapidAPI Tank01 and calculates team impact scores."""

    BASE_URL = "https://tank01-fantasy-stats.p.rapidapi.com"
    HEADERS = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }

    # Team name to abbreviation mapping
    TEAM_ABBREVIATIONS = {
        'Orlando Magic': 'ORL',
        'Minnesota Timberwolves': 'MIN',
        'Miami Heat': 'MIA',
        'Boston Celtics': 'BOS',
        'LA Clippers': 'LAC',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Atlanta Hawks': 'ATL',
        'Cleveland Cavaliers': 'CLE',
        'Toronto Raptors': 'TOR',
        'Washington Wizards': 'WAS',
        'Phoenix Suns': 'PHO',
        'San Antonio Spurs': 'SA',
        'Chicago Bulls': 'CHI',
        'Charlotte Hornets': 'CHA',
        'Philadelphia 76ers': 'PHI',
        'New Orleans Pelicans': 'NO',
        'Sacramento Kings': 'SAC',
        'Dallas Mavericks': 'DAL',
        'Houston Rockets': 'HOU',
        'Brooklyn Nets': 'BKN',
        'New York Knicks': 'NY',
        'Utah Jazz': 'UTA',
        'Oklahoma City Thunder': 'OKC',
        'Portland Trail Blazers': 'POR',
        'Indiana Pacers': 'IND',
        'Milwaukee Bucks': 'MIL',
        'Golden State Warriors': 'GS',
        'Memphis Grizzlies': 'MEM',
        'Los Angeles Lakers': 'LAL'
    }

    # Cache for API responses (team_abv -> (timestamp, data))
    _roster_cache = {}
    _minutes_cache = {}  # player_id -> (timestamp, avg_minutes)
    CACHE_TTL = 600  # 10 minutes

    def __init__(self, teams: list, include_questionable: bool = False):
        """
        Initialize the injury provider.

        Args:
            teams: List of full team names to fetch injuries for
            include_questionable: If True, include Questionable players in impact calculation
        """
        self.teams = teams
        self.include_questionable = include_questionable
        self._injury_data = {}
        self._fetch_all_injuries()

    def _get_team_abbreviation(self, team_name: str) -> str:
        """Convert full team name to abbreviation."""
        return self.TEAM_ABBREVIATIONS.get(team_name)

    def _is_cache_valid(self, cache_entry) -> bool:
        """Check if a cache entry is still valid."""
        if cache_entry is None:
            return False
        timestamp, _ = cache_entry
        return (time.time() - timestamp) < self.CACHE_TTL

    def _fetch_roster(self, team_abv: str) -> dict:
        """Fetch team roster with injury data from API."""
        # Check cache first
        cache_key = team_abv
        if cache_key in self._roster_cache and self._is_cache_valid(self._roster_cache[cache_key]):
            return self._roster_cache[cache_key][1]

        url = f"{self.BASE_URL}/getNBATeamRoster"
        params = {"teamAbv": team_abv}

        try:
            response = requests.get(url, headers=self.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('statusCode') == 200:
                roster = data.get('body', {}).get('roster', [])
                self._roster_cache[cache_key] = (time.time(), roster)
                return roster

        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch roster for {team_abv}: {e}")

        return []

    def _fetch_player_avg_minutes(self, player_id: str) -> float:
        """Fetch average minutes for a player from their last 10 games."""
        # Check cache first
        if player_id in self._minutes_cache and self._is_cache_valid(self._minutes_cache[player_id]):
            return self._minutes_cache[player_id][1]

        url = f"{self.BASE_URL}/getNBAGamesForPlayer"
        params = {
            "playerID": player_id,
            "season": "2024"  # Current season
        }

        try:
            response = requests.get(url, headers=self.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('statusCode') == 200:
                games = list(data.get('body', {}).values())
                # Sort by game ID (most recent first) and take last 10
                games.sort(key=lambda x: x.get('gameID', ''), reverse=True)
                recent_games = games[:10]

                if recent_games:
                    total_minutes = 0
                    games_with_minutes = 0

                    for game in recent_games:
                        mins_str = game.get('mins', '0')
                        try:
                            # Handle "MM:SS" format
                            if ':' in str(mins_str):
                                parts = mins_str.split(':')
                                mins = int(parts[0]) + int(parts[1]) / 60
                            else:
                                mins = float(mins_str)
                            total_minutes += mins
                            games_with_minutes += 1
                        except (ValueError, TypeError):
                            continue

                    if games_with_minutes > 0:
                        avg_minutes = total_minutes / games_with_minutes
                        self._minutes_cache[player_id] = (time.time(), avg_minutes)
                        return avg_minutes

        except requests.RequestException as e:
            # Don't spam warnings for minutes lookups
            pass

        # Default to 0 if we can't get data
        self._minutes_cache[player_id] = (time.time(), 0)
        return 0

    def _fetch_all_injuries(self):
        """Fetch injury data for all teams."""
        for team_name in self.teams:
            team_abv = self._get_team_abbreviation(team_name)
            if not team_abv:
                print(f"  Warning: Unknown team abbreviation for {team_name}")
                continue

            roster = self._fetch_roster(team_abv)
            injured_players = []
            total_impact = 0.0

            for player in roster:
                injury_info = player.get('injury')
                if not injury_info:
                    continue

                designation = injury_info.get('designation', '')
                if not designation:
                    continue

                # Filter based on designation
                is_out = designation.lower() in ['out', 'o']
                is_questionable = designation.lower() in ['questionable', 'q', 'doubtful', 'd', 'day-to-day', 'dtd']

                if not is_out and not (self.include_questionable and is_questionable):
                    continue

                player_id = player.get('playerID')
                player_name = player.get('longName', player.get('shortName', 'Unknown'))

                # Only fetch minutes for players who are OUT (to reduce API calls)
                if is_out and player_id:
                    avg_minutes = self._fetch_player_avg_minutes(player_id)
                else:
                    avg_minutes = 0

                # Calculate importance (minutes / 48)
                importance = avg_minutes / 48.0 if avg_minutes > 0 else 0

                injured_player = {
                    'name': player_name,
                    'designation': designation,
                    'avg_minutes': round(avg_minutes, 1),
                    'importance': round(importance, 3)
                }
                injured_players.append(injured_player)

                # Only count OUT players for impact
                if is_out:
                    total_impact += importance

            self._injury_data[team_name] = {
                'injured_players': injured_players,
                'total_impact': round(total_impact, 3)
            }

    def get_team_injuries(self, team_name: str) -> dict:
        """
        Get injury data for a specific team.

        Args:
            team_name: Full team name (e.g., "Los Angeles Lakers")

        Returns:
            dict with:
                - injured_players: List of injured players with details
                - total_impact: Sum of importance scores for OUT players
        """
        return self._injury_data.get(team_name, {
            'injured_players': [],
            'total_impact': 0.0
        })

    def get_all_injuries(self) -> dict:
        """Get injury data for all teams."""
        return self._injury_data


if __name__ == "__main__":
    # Test the provider
    test_teams = ["Los Angeles Lakers", "Boston Celtics"]
    print(f"Fetching injury data for: {test_teams}\n")

    provider = InjuryProvider(test_teams)

    for team in test_teams:
        injuries = provider.get_team_injuries(team)
        print(f"{team}:")
        print(f"  Total Impact: {injuries['total_impact']}")
        print(f"  Injured Players:")
        for player in injuries['injured_players']:
            print(f"    - {player['name']}: {player['designation']} "
                  f"(avg {player['avg_minutes']} min, importance {player['importance']})")
        print()
