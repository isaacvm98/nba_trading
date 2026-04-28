"""
Logs Polymarket price updates to a CSV file for historical analysis.

Appends a row every time a price is observed (WebSocket or REST).
Deduplicates by skipping updates where price hasn't changed.

Output: Data/paper_trading/price_log.csv
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("Data/paper_trading")
LOG_FILE = DATA_DIR / "price_log.csv"
COLUMNS = ["timestamp", "game_date", "home_team", "away_team", "home_price", "away_price", "source"]


class PriceLogger:
    """Thread-safe price logger that appends to a CSV."""

    def __init__(self, log_file=None):
        self.log_file = log_file or LOG_FILE
        self._lock = threading.Lock()
        self._last_prices = {}  # game_key -> (home_price, away_price)
        self._ensure_file()

    def _ensure_file(self):
        """Create the CSV file with headers if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)

    def log(self, game_date, home_team, away_team, home_price, away_price, source="rest"):
        """Log a price observation. Skips if price unchanged from last update."""
        game_key = f"{game_date}_{home_team}"

        # Round to avoid float noise
        home_price = round(float(home_price), 4)
        away_price = round(float(away_price), 4)

        # Skip if identical to last logged price for this game
        last = self._last_prices.get(game_key)
        if last and last == (home_price, away_price):
            return

        self._last_prices[game_key] = (home_price, away_price)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = [now, game_date, home_team, away_team, home_price, away_price, source]

        with self._lock:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)


# Singleton instance
_logger = None


def get_price_logger():
    """Get the singleton PriceLogger instance."""
    global _logger
    if _logger is None:
        _logger = PriceLogger()
    return _logger
