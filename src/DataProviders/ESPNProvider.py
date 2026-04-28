"""
ESPN API Data Provider

Fetches NBA and CBB data from ESPN's public (unofficial) API endpoints:
- Win probability for live/completed games (play-by-play updates)
- Scoreboard data with game IDs
- Injury data
- Team information

Note: These are undocumented endpoints that may change without notice.
"""

import requests
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from zoneinfo import ZoneInfo

# Eastern Time for game scheduling
ET = ZoneInfo("America/New_York")


class ESPNProvider:
    """Fetches NBA/CBB data from ESPN's public API endpoints."""

    # League-specific base URLs
    LEAGUE_CONFIG = {
        'nba': {
            'site_base': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba',
            'core_base': 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba',
        },
        'cbb': {
            'site_base': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball',
            'core_base': 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball',
        }
    }

    # Default URLs for backwards compatibility (overridden in __init__)
    _DEFAULT_SITE_BASE = LEAGUE_CONFIG['nba']['site_base']
    _DEFAULT_CORE_BASE = LEAGUE_CONFIG['nba']['core_base']

    # ESPN Team ID mapping (ESPN uses numeric IDs)
    # Maps full team name -> ESPN team ID
    TEAM_IDS = {
        'Atlanta Hawks': '1',
        'Boston Celtics': '2',
        'Brooklyn Nets': '17',
        'Charlotte Hornets': '30',
        'Chicago Bulls': '4',
        'Cleveland Cavaliers': '5',
        'Dallas Mavericks': '6',
        'Denver Nuggets': '7',
        'Detroit Pistons': '8',
        'Golden State Warriors': '9',
        'Houston Rockets': '10',
        'Indiana Pacers': '11',
        'LA Clippers': '12',
        'Los Angeles Lakers': '13',
        'Memphis Grizzlies': '29',
        'Miami Heat': '14',
        'Milwaukee Bucks': '15',
        'Minnesota Timberwolves': '16',
        'New Orleans Pelicans': '3',
        'New York Knicks': '18',
        'Oklahoma City Thunder': '25',
        'Orlando Magic': '19',
        'Philadelphia 76ers': '20',
        'Phoenix Suns': '21',
        'Portland Trail Blazers': '22',
        'Sacramento Kings': '23',
        'San Antonio Spurs': '24',
        'Toronto Raptors': '28',
        'Utah Jazz': '26',
        'Washington Wizards': '27'
    }

    # Reverse mapping: ESPN team abbreviation -> full team name
    TEAM_ABBR_TO_NAME = {
        'ATL': 'Atlanta Hawks',
        'BOS': 'Boston Celtics',
        'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls',
        'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks',
        'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors',
        'GS': 'Golden State Warriors',
        'HOU': 'Houston Rockets',
        'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers',
        'LAL': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks',
        'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans',
        'NO': 'New Orleans Pelicans',
        'NYK': 'New York Knicks',
        'NY': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers',
        'PHX': 'Phoenix Suns',
        'PHO': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings',
        'SAS': 'San Antonio Spurs',
        'SA': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors',
        'UTA': 'Utah Jazz',
        'WAS': 'Washington Wizards'
    }

    # Cache for API responses (shared across instances)
    _cache = {}
    CACHE_TTL = 15  # 15 seconds — dashboard refreshes every 5s, ESPN updates every ~15-30s

    def __init__(self, league: str = 'nba'):
        """
        Initialize the ESPN provider.

        Args:
            league: 'nba' or 'cbb' (college basketball)
        """
        self.league = league.lower()
        if self.league not in self.LEAGUE_CONFIG:
            raise ValueError(f"Unknown league: {league}. Use 'nba' or 'cbb'")

        self.site_base = self.LEAGUE_CONFIG[self.league]['site_base']
        self.core_base = self.LEAGUE_CONFIG[self.league]['core_base']

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache:
            return False
        timestamp, _ = self._cache[cache_key]
        return (time.time() - timestamp) < self.CACHE_TTL

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """Cache data with current timestamp."""
        self._cache[cache_key] = (time.time(), data)

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make an HTTP request with error handling."""
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"ESPN API error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Scoreboard Methods
    # -------------------------------------------------------------------------

    def get_scoreboard(self, date: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch NBA scoreboard data for a given date.

        Args:
            date: Date in YYYYMMDD format. If None, fetches today's games.

        Returns:
            dict: Scoreboard data including events (games)
        """
        cache_key = f"scoreboard_{date or 'today'}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{self.site_base}/scoreboard"
        params = {}
        if date:
            params['dates'] = date

        data = self._make_request(url, params)
        if data:
            self._set_cache(cache_key, data)
        return data

    def get_todays_games(self) -> List[Dict]:
        """
        Get list of today's games with basic info.

        Returns:
            List of game dicts with: event_id, competition_id, home_team, away_team,
            status, start_time, scores (if available)
        """
        scoreboard = self.get_scoreboard()
        if not scoreboard:
            return []

        games = []
        events = scoreboard.get('events', [])

        for event in events:
            competition = event.get('competitions', [{}])[0]

            # Parse teams
            competitors = competition.get('competitors', [])
            home_team = None
            away_team = None
            home_score = None
            away_score = None

            for comp in competitors:
                team_data = comp.get('team', {})
                team_abbr = team_data.get('abbreviation', '')
                team_name = self.TEAM_ABBR_TO_NAME.get(team_abbr, team_data.get('displayName', ''))
                score = comp.get('score')

                if comp.get('homeAway') == 'home':
                    home_team = team_name
                    home_score = score
                else:
                    away_team = team_name
                    away_score = score

            # Parse status
            status_data = competition.get('status', {})
            status_type = status_data.get('type', {})
            status = status_type.get('name', 'Unknown')  # e.g., 'STATUS_SCHEDULED', 'STATUS_IN_PROGRESS', 'STATUS_FINAL'

            # Parse start time
            start_time_str = event.get('date', '')
            start_time = None
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                except ValueError:
                    pass

            game_info = {
                'event_id': event.get('id'),
                'competition_id': competition.get('id'),
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status,
                'status_detail': status_data.get('type', {}).get('description', ''),
                'start_time': start_time,
                'venue': competition.get('venue', {}).get('fullName', '')
            }
            games.append(game_info)

        return games

    # -------------------------------------------------------------------------
    # Win Probability Methods (Live Play-by-Play)
    # -------------------------------------------------------------------------

    def get_game_summary(self, event_id: str) -> Optional[Dict]:
        """
        Fetch full game summary including win probability, plays, odds, and boxscore.

        Args:
            event_id: ESPN event ID

        Returns:
            dict: Full game summary data
        """
        cache_key = f"summary_{event_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{self.site_base}/summary"
        params = {'event': event_id}

        data = self._make_request(url, params)
        if data:
            self._set_cache(cache_key, data)
        return data

    def get_live_win_probability(self, event_id: str) -> Optional[Dict]:
        """
        Get ESPN's live play-by-play win probability for a game.

        This is ESPN's statistical model that updates after each play,
        showing the probability of each team winning based on:
        - Current score
        - Time remaining
        - Game situation

        Args:
            event_id: ESPN event ID

        Returns:
            dict with:
                - current: Current win probability snapshot
                - history: Full time series of probability changes
                - plays_count: Number of plays/updates
                - is_live: Whether game is in progress
        """
        summary = self.get_game_summary(event_id)
        if not summary:
            return None

        win_prob = summary.get('winprobability', [])
        plays = summary.get('plays', [])

        if not win_prob:
            return None

        # Build a lookup of play details by ID
        play_details = {str(p.get('id')): p for p in plays}

        # Get current (latest) probability with play context
        latest_wp = win_prob[-1]
        latest_play_id = str(latest_wp.get('playId', ''))
        latest_play = play_details.get(latest_play_id, {})

        current = {
            'home_win_prob': latest_wp.get('homeWinPercentage', 0.5),
            'away_win_prob': 1 - latest_wp.get('homeWinPercentage', 0.5),
            'home_score': latest_play.get('homeScore', 0),
            'away_score': latest_play.get('awayScore', 0),
            'period': latest_play.get('period', {}).get('number', 0),
            'clock': latest_play.get('clock', {}).get('displayValue', ''),
            'last_play': latest_play.get('text', '')
        }

        # Determine game state
        # Check if any play contains "End of Game"
        is_final = any('End of Game' in play_details.get(str(wp.get('playId', '')), {}).get('text', '')
                       for wp in win_prob[-5:])  # Check last few plays

        # Also check for extreme final probabilities (>99% or <1%)
        final_prob = latest_wp.get('homeWinPercentage', 0.5)
        if len(win_prob) > 100 and (final_prob > 0.99 or final_prob < 0.01):
            is_final = True

        is_live = len(win_prob) > 1 and not is_final

        return {
            'current': current,
            'history': win_prob,
            'plays_count': len(win_prob),
            'is_live': is_live,
            'is_final': is_final
        }

    def get_win_probability_at_score(self, event_id: str, home_score: int, away_score: int) -> Optional[Dict]:
        """
        Get the win probability at a specific score in the game.

        Useful for analyzing how probability changed at key moments.

        Args:
            event_id: ESPN event ID
            home_score: Home team score to find
            away_score: Away team score to find

        Returns:
            dict with win probability at that score, or None if not found
        """
        summary = self.get_game_summary(event_id)
        if not summary:
            return None

        plays = summary.get('plays', [])
        win_prob = summary.get('winprobability', [])

        # Build play lookup
        play_details = {str(p.get('id')): p for p in plays}

        # Find the first play matching this score
        for wp in win_prob:
            play_id = str(wp.get('playId', ''))
            play = play_details.get(play_id, {})

            if play.get('homeScore') == home_score and play.get('awayScore') == away_score:
                return {
                    'home_win_prob': wp.get('homeWinPercentage', 0.5),
                    'away_win_prob': 1 - wp.get('homeWinPercentage', 0.5),
                    'period': play.get('period', {}).get('number', 0),
                    'clock': play.get('clock', {}).get('displayValue', ''),
                    'play': play.get('text', '')
                }

        return None

    def get_win_probability_summary(self, event_id: str) -> Optional[Dict]:
        """
        Get a summary of win probability changes throughout a game.

        Returns key moments: start, end of each quarter, and final.

        Args:
            event_id: ESPN event ID

        Returns:
            dict with probability at key moments
        """
        summary = self.get_game_summary(event_id)
        if not summary:
            return None

        plays = summary.get('plays', [])
        win_prob = summary.get('winprobability', [])

        if not win_prob or not plays:
            return None

        # Build play lookup
        play_details = {str(p.get('id')): p for p in plays}

        # Track probability by period
        periods = {}
        for wp in win_prob:
            play_id = str(wp.get('playId', ''))
            play = play_details.get(play_id, {})
            period = play.get('period', {}).get('number', 0)

            if period not in periods:
                periods[period] = {'start': None, 'end': None}

            prob_data = {
                'home_win_prob': wp.get('homeWinPercentage', 0.5),
                'home_score': play.get('homeScore', 0),
                'away_score': play.get('awayScore', 0),
                'clock': play.get('clock', {}).get('displayValue', '')
            }

            if periods[period]['start'] is None:
                periods[period]['start'] = prob_data
            periods[period]['end'] = prob_data

        return {
            'pregame_home_prob': win_prob[0].get('homeWinPercentage', 0.5),
            'final_home_prob': win_prob[-1].get('homeWinPercentage', 0.5),
            'periods': periods,
            'total_plays': len(win_prob)
        }

    def get_all_live_win_probabilities(self, date: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get ESPN's live win probabilities for all games on a date.

        This returns ESPN's play-by-play win probability model data,
        which updates after each play during live games.

        Args:
            date: Date in YYYYMMDD format. If None, fetches today's games.

        Returns:
            dict: {
                "home_team:away_team": {
                    'event_id': str,
                    'home_win_prob': float (ESPN model),
                    'away_win_prob': float (ESPN model),
                    'home_score': int,
                    'away_score': int,
                    'period': int,
                    'clock': str,
                    'is_live': bool,
                    'is_final': bool,
                    'plays_count': int
                }
            }
        """
        scoreboard = self.get_scoreboard(date=date)
        if not scoreboard:
            return {}

        results = {}
        events = scoreboard.get('events', [])

        for event in events:
            event_id = event.get('id')
            competition = event.get('competitions', [{}])[0]

            # Parse teams
            competitors = competition.get('competitors', [])
            home_team = None
            away_team = None

            for comp in competitors:
                team_data = comp.get('team', {})
                team_abbr = team_data.get('abbreviation', '')
                team_name = self.TEAM_ABBR_TO_NAME.get(team_abbr, team_data.get('displayName', ''))

                if comp.get('homeAway') == 'home':
                    home_team = team_name
                else:
                    away_team = team_name

            if not all([event_id, home_team, away_team]):
                continue

            # Fetch live win probability
            live_prob = self.get_live_win_probability(event_id)

            game_key = f"{home_team}:{away_team}"

            if live_prob and live_prob.get('current'):
                current = live_prob['current']
                results[game_key] = {
                    'event_id': event_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win_prob': current.get('home_win_prob'),
                    'away_win_prob': current.get('away_win_prob'),
                    'home_score': current.get('home_score'),
                    'away_score': current.get('away_score'),
                    'period': current.get('period'),
                    'clock': current.get('clock'),
                    'last_play': current.get('last_play'),
                    'is_live': live_prob.get('is_live', False),
                    'is_final': live_prob.get('is_final', False),
                    'plays_count': live_prob.get('plays_count', 0)
                }
            else:
                # Game hasn't started - no win prob data yet
                results[game_key] = {
                    'event_id': event_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_win_prob': None,
                    'away_win_prob': None,
                    'home_score': None,
                    'away_score': None,
                    'period': None,
                    'clock': None,
                    'last_play': None,
                    'is_live': False,
                    'is_final': False,
                    'plays_count': 0
                }

        return results

    # -------------------------------------------------------------------------
    # Injury Methods (Experimental)
    # -------------------------------------------------------------------------

    def get_team_injuries(self, team_name: str) -> Optional[List[Dict]]:
        """
        Attempt to fetch injury data for a team from ESPN.

        Note: This endpoint may not be available or may have different structure.

        Args:
            team_name: Full team name (e.g., "Los Angeles Lakers")

        Returns:
            List of injury records or None if not available
        """
        team_id = self.TEAM_IDS.get(team_name)
        if not team_id:
            print(f"Unknown team: {team_name}")
            return None

        cache_key = f"injuries_{team_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:  # Allow empty list as valid cache
            return cached

        # Try the injuries endpoint (may not work for NBA)
        url = f"{self.core_base}/teams/{team_id}/injuries"

        data = self._make_request(url)
        if data:
            injuries = []
            items = data.get('items', [])

            for item in items:
                # Each item might be a reference that needs to be fetched
                if '$ref' in item:
                    injury_data = self._make_request(item['$ref'])
                    if injury_data:
                        injuries.append(injury_data)
                else:
                    injuries.append(item)

            self._set_cache(cache_key, injuries)
            return injuries

        # Cache empty result to avoid repeated failed requests
        self._set_cache(cache_key, [])
        return []

    def get_all_injuries(self, teams: List[str]) -> Dict[str, List[Dict]]:
        """
        Fetch injuries for multiple teams.

        Args:
            teams: List of full team names

        Returns:
            dict: {team_name: [injury_records]}
        """
        results = {}
        for team in teams:
            injuries = self.get_team_injuries(team)
            results[team] = injuries if injuries else []
        return results

    # -------------------------------------------------------------------------
    # Team & Roster Methods
    # -------------------------------------------------------------------------

    def get_teams(self) -> Optional[Dict]:
        """
        Get list of all NBA teams.

        Returns:
            dict: Teams data from ESPN
        """
        cache_key = "teams"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{self.site_base}/teams"
        data = self._make_request(url)
        if data:
            self._set_cache(cache_key, data)
        return data

    def get_team_roster(self, team_name: str) -> Optional[Dict]:
        """
        Get roster for a specific team.

        Args:
            team_name: Full team name

        Returns:
            dict: Roster data
        """
        team_id = self.TEAM_IDS.get(team_name)
        if not team_id:
            return None

        cache_key = f"roster_{team_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{self.site_base}/teams/{team_id}/roster"
        data = self._make_request(url)
        if data:
            self._set_cache(cache_key, data)
        return data

    # -------------------------------------------------------------------------
    # Standings Methods
    # -------------------------------------------------------------------------

    def get_standings(self) -> Optional[Dict]:
        """
        Get current NBA standings.

        Returns:
            dict: Standings data from ESPN
        """
        cache_key = "standings"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{self.site_base}/standings"
        data = self._make_request(url)
        if data:
            self._set_cache(cache_key, data)
        return data


if __name__ == "__main__":
    # Test the provider
    provider = ESPNProvider()

    print("=" * 60)
    print("ESPN NBA API Test")
    print("=" * 60)

    # Test scoreboard
    print("\n1. Today's Games:")
    print("-" * 40)
    games = provider.get_todays_games()
    if games:
        for game in games:
            home = game['home_team']
            away = game['away_team']
            status = game['status_detail']
            print(f"  {away} @ {home}")
            print(f"    Status: {status}")
            print(f"    Event ID: {game['event_id']}")
            if game['home_score'] and game['away_score']:
                print(f"    Score: {away} {game['away_score']} - {home} {game['home_score']}")
            print()
    else:
        print("  No games found for today")

    # Test win probability
    print("\n2. Win Probabilities:")
    print("-" * 40)
    probs = provider.get_all_live_win_probabilities()
    if probs:
        for game_key, data in probs.items():
            home = data['home_team']
            away = data['away_team']
            home_prob = data.get('home_team_prob')
            away_prob = data.get('away_team_prob')

            print(f"  {away} @ {home}")
            if home_prob is not None:
                print(f"    {home}: {home_prob:.1%}")
                print(f"    {away}: {away_prob:.1%}")
            else:
                print(f"    Win probability not available")
            print(f"    Status: {data.get('status_detail')}")
            print(f"    Live: {data.get('is_live')}")
            print()
    else:
        print("  No probability data available")

    # Test injury endpoint (experimental)
    print("\n3. Injury Data (Experimental):")
    print("-" * 40)
    test_teams = ["Los Angeles Lakers", "Boston Celtics"]
    for team in test_teams:
        injuries = provider.get_team_injuries(team)
        print(f"  {team}:")
        if injuries:
            for injury in injuries[:3]:  # Show first 3
                print(f"    - {injury}")
        else:
            print(f"    No injury data available (endpoint may not exist for NBA)")
        print()

    print("=" * 60)
    print("Test complete")
    print("=" * 60)
