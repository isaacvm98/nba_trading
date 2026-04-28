"""
Logs ESPN win probability updates to a CSV file for backtest analysis.

Appends a row every time a WP observation changes for a game.
Deduplicates by skipping updates where WP and score haven't changed.

Output: Data/paper_trading/espn_wp_log.csv
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("Data/paper_trading")
LOG_FILE = DATA_DIR / "espn_wp_log.csv"
COLUMNS = [
    "timestamp", "game_date", "home_team", "away_team",
    "home_wp", "away_wp", "home_score", "away_score",
    "period", "clock", "source",
]


class ESPNWPLogger:
    """Thread-safe ESPN win probability logger that appends to a CSV."""

    def __init__(self, log_file=None):
        self.log_file = log_file or LOG_FILE
        self._lock = threading.Lock()
        self._last_state = {}  # game_key -> (home_wp, away_wp, home_score, away_score)
        self._ensure_file()

    def _ensure_file(self):
        """Create the CSV file with headers if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)

    def log(self, game_date, home_team, away_team, home_wp, away_wp,
            home_score=None, away_score=None, period=None, clock=None,
            source="scheduler"):
        """Log an ESPN WP observation. Skips if state unchanged from last update."""
        game_key = f"{game_date}_{home_team}"

        home_wp = round(float(home_wp), 4) if home_wp is not None else None
        away_wp = round(float(away_wp), 4) if away_wp is not None else None

        if home_wp is None:
            return

        state = (home_wp, away_wp, home_score, away_score)

        # Skip if identical to last logged state for this game
        last = self._last_state.get(game_key)
        if last and last == state:
            return

        self._last_state[game_key] = state

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = [
            now, game_date, home_team, away_team,
            home_wp, away_wp, home_score, away_score,
            period, clock or "", source,
        ]

        with self._lock:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)


# Singleton instance
_logger = None


def get_espn_wp_logger():
    """Get the singleton ESPNWPLogger instance."""
    global _logger
    if _logger is None:
        _logger = ESPNWPLogger()
    return _logger
