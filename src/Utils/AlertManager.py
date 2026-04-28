"""
Alert Manager for NBA Betting System

Handles notifications for:
- Drawdown limit warnings/breaches
- Position resolutions (wins/losses)
- System status changes
- Daily summaries

Supports multiple notification channels:
- Console (always)
- File logging (default)
- Future: Slack, Email, Discord, etc.

Usage:
    from src.Utils.AlertManager import AlertManager, AlertType

    am = AlertManager()
    am.alert(AlertType.WARNING, "Approaching daily limit", {"daily_pnl": -45})
    am.alert(AlertType.RESOLUTION, "BOS @ NYK resolved", {"won": True, "pnl": 50})
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable


class AlertType(Enum):
    """Types of alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    ENTRY = "ENTRY"
    RESOLUTION = "RESOLUTION"
    DRAWDOWN = "DRAWDOWN"
    HALT = "HALT"
    DAILY_SUMMARY = "DAILY_SUMMARY"


class AlertChannel:
    """Base class for alert channels."""

    def send(self, alert_type: AlertType, message: str, data: Dict[str, Any] = None):
        """Send an alert through this channel."""
        raise NotImplementedError


class ConsoleChannel(AlertChannel):
    """Send alerts to console with color coding."""

    COLORS = {
        AlertType.INFO: "\033[94m",      # Blue
        AlertType.WARNING: "\033[93m",   # Yellow
        AlertType.ERROR: "\033[91m",     # Red
        AlertType.RESOLUTION: "\033[92m", # Green
        AlertType.DRAWDOWN: "\033[93m",  # Yellow
        AlertType.HALT: "\033[91m",      # Red
        AlertType.DAILY_SUMMARY: "\033[96m",  # Cyan
    }
    RESET = "\033[0m"

    def send(self, alert_type: AlertType, message: str, data: Dict[str, Any] = None):
        color = self.COLORS.get(alert_type, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{alert_type.value}]"

        print(f"{color}{timestamp} {prefix} {message}{self.RESET}")

        if data:
            for key, value in data.items():
                print(f"  {key}: {value}")


class FileChannel(AlertChannel):
    """Log alerts to a file."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert_type: AlertType, message: str, data: Dict[str, Any] = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": alert_type.value,
            "message": message,
            "data": data or {},
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")


class WebhookChannel(AlertChannel):
    """Send alerts to a webhook URL (Slack, Discord, etc.)."""

    def __init__(self, webhook_url: str, platform: str = "discord"):
        self.webhook_url = webhook_url
        self.platform = platform.lower()

    def _format_discord(self, alert_type: AlertType, message: str, data: Dict) -> Dict:
        """Format payload for Discord webhooks."""
        # Discord color codes (decimal)
        color_map = {
            AlertType.INFO: 3447003,       # Blue
            AlertType.WARNING: 16776960,   # Yellow
            AlertType.ERROR: 15158332,     # Red
            AlertType.ENTRY: 10181046,     # Purple
            AlertType.RESOLUTION: 3066993, # Green
            AlertType.DRAWDOWN: 16776960,  # Yellow
            AlertType.HALT: 15158332,      # Red
            AlertType.DAILY_SUMMARY: 1752220,  # Teal
        }

        emoji_map = {
            AlertType.INFO: "ℹ️",
            AlertType.WARNING: "⚠️",
            AlertType.ERROR: "❌",
            AlertType.ENTRY: "🎯",
            AlertType.RESOLUTION: "💰",
            AlertType.DRAWDOWN: "📉",
            AlertType.HALT: "🛑",
            AlertType.DAILY_SUMMARY: "📊",
        }

        emoji = emoji_map.get(alert_type, "🔔")
        color = color_map.get(alert_type, 0)

        # Build embed
        embed = {
            "title": f"{emoji} {alert_type.value}",
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add fields for data
        if data:
            fields = []
            for key, value in data.items():
                # Format value nicely
                if isinstance(value, float):
                    if 'pnl' in key.lower() or 'bankroll' in key.lower():
                        value = f"${value:+.2f}" if 'pnl' in key.lower() else f"${value:.2f}"
                    else:
                        value = f"{value:.2f}"
                fields.append({
                    "name": key.replace('_', ' ').title(),
                    "value": str(value),
                    "inline": True
                })
            embed["fields"] = fields

        return {
            "embeds": [embed],
            "username": "NBA Betting Bot"
        }

    def _format_slack(self, alert_type: AlertType, message: str, data: Dict) -> Dict:
        """Format payload for Slack webhooks."""
        emoji_map = {
            AlertType.INFO: ":information_source:",
            AlertType.WARNING: ":warning:",
            AlertType.ERROR: ":x:",
            AlertType.ENTRY: ":dart:",
            AlertType.RESOLUTION: ":moneybag:",
            AlertType.DRAWDOWN: ":chart_with_downwards_trend:",
            AlertType.HALT: ":octagonal_sign:",
            AlertType.DAILY_SUMMARY: ":bar_chart:",
        }

        emoji = emoji_map.get(alert_type, ":bell:")
        text = f"{emoji} *{alert_type.value}*: {message}"

        if data:
            text += "\n```\n"
            for key, value in data.items():
                text += f"{key}: {value}\n"
            text += "```"

        return {"text": text}

    def send(self, alert_type: AlertType, message: str, data: Dict[str, Any] = None):
        try:
            import urllib.request

            if self.platform == "discord":
                payload = self._format_discord(alert_type, message, data or {})
            else:
                payload = self._format_slack(alert_type, message, data or {})

            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'NBA-Betting-Bot/2.0'
                }
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            # Don't fail silently but don't crash either
            print(f"[AlertManager] Webhook failed: {e}")


class AlertManager:
    """Manages alerts across multiple channels."""

    def __init__(
        self,
        data_dir: Path = Path("Data/paper_trading"),
        enable_console: bool = True,
        enable_file: bool = True,
        webhook_url: str = None,
        webhook_platform: str = "discord",
    ):
        self.data_dir = Path(data_dir)
        self.channels: List[AlertChannel] = []

        if enable_console:
            self.channels.append(ConsoleChannel())

        if enable_file:
            log_path = self.data_dir / "alerts.jsonl"
            self.channels.append(FileChannel(log_path))

        if webhook_url:
            self.channels.append(WebhookChannel(webhook_url, platform=webhook_platform))

        # Track alerts to avoid duplicates
        self._recent_alerts: Dict[str, datetime] = {}
        self._dedup_window_seconds = 60  # Don't repeat same alert within 60 seconds

    def add_channel(self, channel: AlertChannel):
        """Add a custom alert channel."""
        self.channels.append(channel)

    def alert(
        self,
        alert_type: AlertType,
        message: str,
        data: Dict[str, Any] = None,
        dedupe_key: str = None
    ):
        """Send an alert through all channels.

        Args:
            alert_type: Type of alert
            message: Alert message
            data: Optional additional data
            dedupe_key: Optional key for deduplication
        """
        # Check for duplicate
        if dedupe_key:
            if dedupe_key in self._recent_alerts:
                last_time = self._recent_alerts[dedupe_key]
                elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
                if elapsed < self._dedup_window_seconds:
                    return  # Skip duplicate

            self._recent_alerts[dedupe_key] = datetime.now(timezone.utc)

        # Send to all channels
        for channel in self.channels:
            try:
                channel.send(alert_type, message, data)
            except Exception as e:
                print(f"[AlertManager] Channel error: {e}")

    def info(self, message: str, data: Dict[str, Any] = None):
        """Send an info alert."""
        self.alert(AlertType.INFO, message, data)

    def warning(self, message: str, data: Dict[str, Any] = None):
        """Send a warning alert."""
        self.alert(AlertType.WARNING, message, data)

    def error(self, message: str, data: Dict[str, Any] = None):
        """Send an error alert."""
        self.alert(AlertType.ERROR, message, data)

    def resolution(self, game: str, won: bool, pnl: float, data: Dict[str, Any] = None):
        """Send a position resolution alert."""
        result = "WON" if won else "LOST"
        message = f"{game}: {result} ${abs(pnl):.2f}"
        alert_data = {"won": won, "pnl": pnl}
        if data:
            alert_data.update(data)
        self.alert(AlertType.RESOLUTION, message, alert_data)

    def entry(self, game: str, bet_side: str, entry_price: float, bet_amount: float, edge: float, data: Dict[str, Any] = None):
        """Send a bet entry alert."""
        message = f"{game}: {bet_side.upper()} @ {entry_price:.1%} | ${bet_amount:.2f} | Edge: {edge:+.1f}"
        alert_data = {"bet_side": bet_side, "entry_price": entry_price, "bet_amount": bet_amount, "edge": edge}
        if data:
            alert_data.update(data)
        self.alert(AlertType.ENTRY, message, alert_data)

    def drawdown_warning(self, limit_type: str, current_pct: float, limit_pct: float):
        """Send a drawdown warning alert."""
        message = f"{limit_type} at {current_pct:.0%} of {limit_pct:.0%} limit"
        self.alert(
            AlertType.DRAWDOWN,
            message,
            {"limit_type": limit_type, "current_pct": current_pct, "limit_pct": limit_pct},
            dedupe_key=f"drawdown_{limit_type}"
        )

    def trading_halted(self, reason: str, data: Dict[str, Any] = None):
        """Send a trading halted alert."""
        self.alert(AlertType.HALT, f"Trading HALTED: {reason}", data)

    def daily_summary(
        self,
        total_pnl: float,
        wins: int,
        losses: int,
        open_positions: int,
        data: Dict[str, Any] = None
    ):
        """Send a daily summary alert."""
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        message = f"P&L: ${total_pnl:+.2f} | {wins}W-{losses}L ({win_rate:.0f}%) | {open_positions} open"
        self.alert(AlertType.DAILY_SUMMARY, message, data)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts from the log file.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts, newest first
        """
        log_path = self.data_dir / "alerts.jsonl"
        if not log_path.exists():
            return []

        alerts = []
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    alerts.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Return newest first, limited
        return list(reversed(alerts[-limit:]))


def main():
    """Demo the alert manager."""
    import argparse

    parser = argparse.ArgumentParser(description='Alert Manager')
    parser.add_argument('--test', action='store_true', help='Send test alerts')
    parser.add_argument('--recent', type=int, default=10, help='Show recent alerts')

    args = parser.parse_args()

    am = AlertManager()

    if args.test:
        print("Sending test alerts...\n")
        am.info("System started", {"version": "1.0"})
        am.warning("Approaching daily limit", {"daily_pnl": -45, "limit": -50})
        am.resolution("BOS @ NYK", won=True, pnl=75.50)
        am.resolution("LAL @ GSW", won=False, pnl=-50.00)
        am.drawdown_warning("Daily", 0.85, 1.0)
        am.daily_summary(total_pnl=25.50, wins=3, losses=2, open_positions=2)
        print("\nTest alerts sent!")
    else:
        # Show recent alerts
        alerts = am.get_recent_alerts(args.recent)
        if alerts:
            print(f"Recent {len(alerts)} alerts:")
            print("-" * 50)
            for alert in alerts:
                ts = alert['timestamp'][:19].replace('T', ' ')
                print(f"{ts} [{alert['type']}] {alert['message']}")
        else:
            print("No recent alerts found.")


if __name__ == "__main__":
    main()
