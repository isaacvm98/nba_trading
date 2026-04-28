"""
ESPN Live Game Tracker

Efficiently tracks live games and updates win probability only when new plays occur.
Uses smart polling to minimize API calls while keeping data fresh.
"""

import time
import threading
from typing import Dict, Optional, List, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from src.DataProviders.ESPNProvider import ESPNProvider

ET = ZoneInfo("America/New_York")


class ESPNLiveTracker:
    """
    Tracks live games and fires callbacks when win probability changes.

    Features:
    - Polls ESPN every N seconds during live games
    - Only triggers updates when new plays are detected
    - Automatically stops tracking when games end
    - Minimal API calls when no games are live
    """

    def __init__(self, league: str = 'nba', poll_interval: int = 15):
        """
        Initialize the live tracker.

        Args:
            league: 'nba' or 'cbb'
            poll_interval: Seconds between polls during live games (default 15)
        """
        self.league = league
        self.poll_interval = poll_interval
        self.provider = ESPNProvider(league)

        # Disable provider caching - we manage our own
        self.provider.CACHE_TTL = 0

        # Track game states
        self._game_states: Dict[str, dict] = {}  # event_id -> {plays_count, last_prob, ...}
        self._tracked_games: set = set()  # event_ids we're actively tracking

        # Callbacks
        self._on_probability_change: Optional[Callable] = None
        self._on_game_end: Optional[Callable] = None

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def track_game(self, event_id: str):
        """Start tracking a specific game."""
        with self._lock:
            self._tracked_games.add(event_id)

    def untrack_game(self, event_id: str):
        """Stop tracking a specific game."""
        with self._lock:
            self._tracked_games.discard(event_id)
            self._game_states.pop(event_id, None)

    def track_games_for_positions(self, positions: List[dict], espn_data: Dict[str, dict]):
        """
        Track games that have open positions.

        Args:
            positions: List of position dicts with 'game' key
            espn_data: Dict from get_espn_live_probabilities with event_ids
        """
        # Find event_ids for open positions
        for game_key, data in espn_data.items():
            event_id = data.get('event_id')
            if event_id and data.get('is_live'):
                self.track_game(event_id)

    def on_probability_change(self, callback: Callable):
        """
        Register callback for probability changes.

        Callback receives: (event_id, game_data) where game_data includes:
            - home_team, away_team
            - home_win_prob, away_win_prob
            - home_score, away_score
            - period, clock
            - last_play
            - plays_count
        """
        self._on_probability_change = callback

    def on_game_end(self, callback: Callable):
        """Register callback for when a game ends."""
        self._on_game_end = callback

    def get_current_state(self, event_id: str) -> Optional[dict]:
        """Get current state for a tracked game."""
        with self._lock:
            return self._game_states.get(event_id)

    def get_all_states(self) -> Dict[str, dict]:
        """Get states for all tracked games."""
        with self._lock:
            return dict(self._game_states)

    def _poll_game(self, event_id: str) -> Optional[dict]:
        """Poll a single game for updates."""
        try:
            live_prob = self.provider.get_live_win_probability(event_id)
            if not live_prob:
                return None

            current = live_prob.get('current', {})

            return {
                'event_id': event_id,
                'home_win_prob': current.get('home_win_prob'),
                'away_win_prob': current.get('away_win_prob'),
                'home_score': current.get('home_score'),
                'away_score': current.get('away_score'),
                'period': current.get('period'),
                'clock': current.get('clock'),
                'last_play': current.get('last_play'),
                'plays_count': live_prob.get('plays_count', 0),
                'is_live': live_prob.get('is_live', False),
                'is_final': live_prob.get('is_final', False),
                'probability_history': [h.get('homeWinPercentage') for h in live_prob.get('history', [])[-20:]],
                'updated_at': datetime.now(ET).isoformat(),
            }
        except Exception as e:
            print(f"Error polling game {event_id}: {e}")
            return None

    def _check_for_updates(self):
        """Check all tracked games for updates."""
        with self._lock:
            games_to_check = list(self._tracked_games)

        for event_id in games_to_check:
            new_state = self._poll_game(event_id)
            if not new_state:
                continue

            with self._lock:
                old_state = self._game_states.get(event_id, {})
                old_plays = old_state.get('plays_count', 0)
                new_plays = new_state.get('plays_count', 0)

                # Check if there's a new play
                if new_plays > old_plays:
                    self._game_states[event_id] = new_state

                    # Fire callback
                    if self._on_probability_change:
                        try:
                            self._on_probability_change(event_id, new_state)
                        except Exception as e:
                            print(f"Callback error: {e}")

                # Check if game ended
                if new_state.get('is_final') and not old_state.get('is_final'):
                    self._game_states[event_id] = new_state
                    self._tracked_games.discard(event_id)

                    if self._on_game_end:
                        try:
                            self._on_game_end(event_id, new_state)
                        except Exception as e:
                            print(f"Game end callback error: {e}")

    def _run_loop(self):
        """Main polling loop."""
        while self._running:
            if self._tracked_games:
                self._check_for_updates()
            time.sleep(self.poll_interval)

    def start(self):
        """Start the live tracker in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"ESPN Live Tracker started (polling every {self.poll_interval}s)")

    def stop(self):
        """Stop the live tracker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("ESPN Live Tracker stopped")

    def poll_once(self) -> Dict[str, dict]:
        """
        Poll all tracked games once (synchronous).

        Returns dict of event_id -> game_state for games with updates.
        """
        updates = {}

        with self._lock:
            games_to_check = list(self._tracked_games)

        for event_id in games_to_check:
            new_state = self._poll_game(event_id)
            if not new_state:
                continue

            with self._lock:
                old_state = self._game_states.get(event_id, {})
                old_plays = old_state.get('plays_count', 0)
                new_plays = new_state.get('plays_count', 0)

                if new_plays > old_plays or not old_state:
                    self._game_states[event_id] = new_state
                    updates[event_id] = new_state

        return updates


# Singleton instances for NBA and CBB
_nba_tracker: Optional[ESPNLiveTracker] = None
_cbb_tracker: Optional[ESPNLiveTracker] = None


def get_live_tracker(league: str = 'nba') -> ESPNLiveTracker:
    """Get or create the singleton live tracker for a league."""
    global _nba_tracker, _cbb_tracker

    if league.lower() == 'nba':
        if _nba_tracker is None:
            _nba_tracker = ESPNLiveTracker('nba')
        return _nba_tracker
    else:
        if _cbb_tracker is None:
            _cbb_tracker = ESPNLiveTracker('cbb')
        return _cbb_tracker


if __name__ == "__main__":
    # Demo the live tracker
    print("=== ESPN Live Tracker Demo ===")
    print()

    tracker = ESPNLiveTracker('nba', poll_interval=10)

    # Define callback
    def on_update(event_id, data):
        print(f"\n🔄 UPDATE: {data.get('last_play', '')[:50]}")
        print(f"   Score: {data['away_score']}-{data['home_score']}")
        print(f"   Home Win: {data['home_win_prob']:.1%}")
        print(f"   Plays: {data['plays_count']}")

    tracker.on_probability_change(on_update)

    # Get today's games
    provider = ESPNProvider('nba')
    games = provider.get_all_live_win_probabilities()

    print(f"Found {len(games)} games")

    # Track live games
    for key, data in games.items():
        if data.get('is_live'):
            print(f"Tracking: {key} (Event ID: {data['event_id']})")
            tracker.track_game(data['event_id'])

    if tracker._tracked_games:
        print(f"\nStarting tracker... (Ctrl+C to stop)")
        tracker.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            tracker.stop()
    else:
        print("No live games to track")
