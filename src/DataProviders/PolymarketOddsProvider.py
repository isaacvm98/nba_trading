import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Eastern Time for NBA game scheduling
ET = ZoneInfo("America/New_York")


class PolymarketOddsProvider:
    """Fetches NBA game odds from Polymarket prediction markets.

    Polymarket returns probability prices (0.0-1.0) which are converted
    to American odds format for compatibility with the existing system.
    """

    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    # NBA series_id and tag_id for game markets (not futures)
    NBA_SERIES_ID = "10345"
    GAMES_TAG_ID = "100639"

    # Mapping of Polymarket team names to the full names used in this project
    TEAM_NAME_MAP = {
        # Short names to full names
        "Lakers": "Los Angeles Lakers",
        "Clippers": "LA Clippers",
        "Warriors": "Golden State Warriors",
        "Kings": "Sacramento Kings",
        "Suns": "Phoenix Suns",
        "Nuggets": "Denver Nuggets",
        "Thunder": "Oklahoma City Thunder",
        "Blazers": "Portland Trail Blazers",
        "Trail Blazers": "Portland Trail Blazers",
        "Jazz": "Utah Jazz",
        "Timberwolves": "Minnesota Timberwolves",
        "Pelicans": "New Orleans Pelicans",
        "Spurs": "San Antonio Spurs",
        "Rockets": "Houston Rockets",
        "Mavericks": "Dallas Mavericks",
        "Grizzlies": "Memphis Grizzlies",
        "Celtics": "Boston Celtics",
        "Nets": "Brooklyn Nets",
        "Knicks": "New York Knicks",
        "76ers": "Philadelphia 76ers",
        "Sixers": "Philadelphia 76ers",
        "Raptors": "Toronto Raptors",
        "Bulls": "Chicago Bulls",
        "Cavaliers": "Cleveland Cavaliers",
        "Cavs": "Cleveland Cavaliers",
        "Pistons": "Detroit Pistons",
        "Pacers": "Indiana Pacers",
        "Bucks": "Milwaukee Bucks",
        "Hawks": "Atlanta Hawks",
        "Hornets": "Charlotte Hornets",
        "Heat": "Miami Heat",
        "Magic": "Orlando Magic",
        "Wizards": "Washington Wizards",
    }

    def __init__(self, sportsbook="polymarket", minutes_before_game=None):
        """Initialize the provider.

        Args:
            sportsbook: Name of the sportsbook (always "polymarket")
            minutes_before_game: If set, only return games starting within this many minutes.
                                 None means return all active games for today.
        """
        self.sportsbook = sportsbook
        self.minutes_before_game = minutes_before_game
        self.events = []
        self._fetch_todays_games()

    def _fetch_todays_games(self):
        """Fetch today's NBA game events from Polymarket Gamma API.

        Uses startTime field which is the actual game start time.
        """
        try:
            params = {
                "series_id": self.NBA_SERIES_ID,
                "tag_id": self.GAMES_TAG_ID,
                "active": "true",
                "closed": "false",
                "order": "startTime",
                "ascending": "true",
                "limit": 100
            }

            response = requests.get(
                f"{self.GAMMA_API_URL}/events",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            all_events = response.json()

            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(ET)
            today_et = now_et.date()

            for event in all_events:
                start_time_str = event.get("startTime", "")
                if not start_time_str:
                    continue

                try:
                    start_time_utc = datetime.fromisoformat(
                        start_time_str.replace("Z", "+00:00")
                    )
                    start_time_et = start_time_utc.astimezone(ET)
                except (ValueError, TypeError):
                    continue

                # Skip games that have already started
                if start_time_utc <= now_utc:
                    continue

                # Only include today's games (ET)
                if start_time_et.date() != today_et:
                    continue

                # If minutes_before_game is set, filter by time window
                if self.minutes_before_game is not None:
                    time_until_start = (start_time_utc - now_utc).total_seconds() / 60
                    # Only include games starting within the window
                    if time_until_start > self.minutes_before_game:
                        continue

                self.events.append(event)

        except requests.RequestException as e:
            print(f"Error fetching Polymarket data: {e}")
            self.events = []

    def _normalize_team_name(self, name):
        """Normalize Polymarket team name to full team name."""
        name = name.strip()
        return self.TEAM_NAME_MAP.get(name, name)

    def _extract_teams_from_event(self, event):
        """Extract home and away team names from an event.

        Polymarket slugs follow pattern: nba-{away}-{home}-{date}
        e.g., nba-den-mem-2026-01-25 = Nuggets @ Grizzlies
        """
        title = event.get("title", "")
        markets = event.get("markets", [])

        # Find moneyline market (Team vs Team, not O/U or Spread)
        for market in markets:
            question = market.get("question", "")
            outcomes = market.get("outcomes", "")

            # Skip O/U, Spread, and 1H (first half) markets
            if "O/U" in question or "Spread" in question or "Over" in outcomes or "1H" in question:
                continue

            # Check if it's a team vs team market
            if " vs. " in question or " vs " in question:
                # Parse outcomes for team names
                if isinstance(outcomes, str):
                    try:
                        import json
                        outcomes = json.loads(outcomes)
                    except:
                        outcomes = outcomes.replace("[", "").replace("]", "").replace('"', "").split(",")

                if len(outcomes) >= 2:
                    # First team is usually away, second is home based on " vs. " convention
                    # But Polymarket uses "Away vs. Home" format
                    away_team = self._normalize_team_name(outcomes[0].strip())
                    home_team = self._normalize_team_name(outcomes[1].strip())
                    return home_team, away_team, market

        return None, None, None

    def _get_moneyline_prices(self, market):
        """Get moneyline probabilities from market."""
        if not market:
            return None, None

        outcome_prices = market.get("outcomePrices", "")

        if isinstance(outcome_prices, str):
            try:
                import json
                outcome_prices = json.loads(outcome_prices)
            except:
                outcome_prices = outcome_prices.replace("[", "").replace("]", "").replace('"', "").split(",")

        if len(outcome_prices) >= 2:
            try:
                # First price is away team, second is home team
                away_prob = float(outcome_prices[0])
                home_prob = float(outcome_prices[1])

                # Filter out resolved/dead markets (extreme probabilities)
                # Real pre-game odds rarely exceed 95%/5%
                if away_prob > 0.95 or away_prob < 0.05:
                    return None, None
                if home_prob > 0.95 or home_prob < 0.05:
                    return None, None

                return home_prob, away_prob
            except (ValueError, TypeError):
                pass

        return None, None

    def _get_clob_token_ids(self, market):
        """Get CLOB token IDs for price history lookup."""
        if not market:
            return None, None

        token_ids = market.get("clobTokenIds", "")

        if isinstance(token_ids, str):
            try:
                import json
                token_ids = json.loads(token_ids)
            except:
                return None, None

        if len(token_ids) >= 2:
            # First token is away team, second is home team
            return token_ids[1], token_ids[0]  # Return (home_token, away_token)

        return None, None

    def _get_over_under_line(self, event):
        """Get the over/under line from an event's markets."""
        import re
        markets = event.get("markets", [])

        for market in markets:
            question = market.get("question", "")

            # Look for game total O/U market (not player props, not 1H)
            # Pattern: "Team vs. Team: O/U 225.5" (not "Player: Something O/U X")
            # Game totals are typically 200-280, player props are much lower
            if "O/U" in question and "1H" not in question and " vs" in question.lower():
                match = re.search(r'O/U\s+(\d+\.?\d*)', question)
                if match:
                    try:
                        value = float(match.group(1))
                        # Game totals are typically between 200-280
                        if 180 <= value <= 300:
                            return value
                    except ValueError:
                        pass

        return None

    def _probability_to_american_odds(self, prob):
        """Convert probability (0-1) to American odds.

        Args:
            prob: Probability between 0 and 1

        Returns:
            American odds (negative for favorites, positive for underdogs)
        """
        if prob is None or prob <= 0 or prob >= 1:
            return None

        if prob >= 0.5:
            # Favorite: negative odds
            return int(round(-100 * prob / (1 - prob)))
        else:
            # Underdog: positive odds
            return int(round(100 * (1 - prob) / prob))

    def get_odds(self):
        """Get odds for today's NBA games in the standard format.

        Returns:
            dict: {
                "home_team:away_team": {
                    "under_over_odds": float or None,
                    "home_team": {"money_line_odds": int},
                    "away_team": {"money_line_odds": int}
                }
            }
        """
        dict_res = {}

        for event in self.events:
            home_team, away_team, ml_market = self._extract_teams_from_event(event)

            if not home_team or not away_team:
                continue

            # Get moneyline probabilities
            home_prob, away_prob = self._get_moneyline_prices(ml_market)

            # Skip games with invalid/resolved odds
            if home_prob is None or away_prob is None:
                continue

            # Convert to American odds
            home_odds = self._probability_to_american_odds(home_prob)
            away_odds = self._probability_to_american_odds(away_prob)

            # Get over/under line
            ou_line = self._get_over_under_line(event)

            # Get CLOB token IDs for price history
            home_token, away_token = self._get_clob_token_ids(ml_market)

            game_key = f"{home_team}:{away_team}"
            dict_res[game_key] = {
                'under_over_odds': ou_line,
                'home_token_id': home_token,
                'away_token_id': away_token,
                home_team: {'money_line_odds': home_odds},
                away_team: {'money_line_odds': away_odds}
            }

        return dict_res


if __name__ == "__main__":
    # Test the provider
    provider = PolymarketOddsProvider()
    odds = provider.get_odds()

    print(f"Found {len(odds)} games\n")
    for game_key, game_odds in odds.items():
        home, away = game_key.split(":")
        print(f"{away} @ {home}")
        print(f"  Home ML: {game_odds[home]['money_line_odds']}")
        print(f"  Away ML: {game_odds[away]['money_line_odds']}")
        print(f"  O/U: {game_odds['under_over_odds']}")
        print()
