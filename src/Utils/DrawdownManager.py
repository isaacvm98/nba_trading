"""
Drawdown Manager for Bankroll-Level Risk Management

Since per-position stop-losses are disabled (they destroy edge),
this module provides bankroll-level protection:
- Daily loss limits
- Weekly loss limits
- Total drawdown limits

Usage:
    from src.Utils.DrawdownManager import DrawdownManager

    dm = DrawdownManager()
    if dm.can_trade():
        # Place bet
        ...
        dm.record_pnl(pnl, position_id)
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


class DrawdownManager:
    """Manages bankroll-level risk by tracking drawdowns and enforcing limits."""

    # Default limits (can be overridden in constructor)
    DEFAULT_MAX_DAILY_LOSS = 1     # 5% of bankroll -> stop trading for day
    DEFAULT_MAX_WEEKLY_LOSS = 1    # 10% of bankroll -> stop trading for week
    DEFAULT_MAX_TOTAL_DRAWDOWN = 0.80  # 20% of bankroll -> halt strategy entirely
    DEFAULT_ALERT_THRESHOLD = 0.80     # Alert when 80% of any limit is reached

    def __init__(
        self,
        data_dir: Path = Path("Data/paper_trading"),
        starting_bankroll: float = 1000.0,
        max_daily_loss: float = None,
        max_weekly_loss: float = None,
        max_total_drawdown: float = None,
        alert_threshold: float = None,
    ):
        """Initialize DrawdownManager.

        Args:
            data_dir: Directory for storing drawdown tracking data
            starting_bankroll: Initial bankroll for calculating drawdown %
            max_daily_loss: Max daily loss as fraction of bankroll (e.g., 0.05 = 5%)
            max_weekly_loss: Max weekly loss as fraction of bankroll
            max_total_drawdown: Max total drawdown from peak as fraction
            alert_threshold: Fraction of limit to trigger warning (e.g., 0.80 = 80%)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.drawdown_file = self.data_dir / "drawdown_state.json"

        self.starting_bankroll = starting_bankroll
        self.max_daily_loss = max_daily_loss or self.DEFAULT_MAX_DAILY_LOSS
        self.max_weekly_loss = max_weekly_loss or self.DEFAULT_MAX_WEEKLY_LOSS
        self.max_total_drawdown = max_total_drawdown or self.DEFAULT_MAX_TOTAL_DRAWDOWN
        self.alert_threshold = alert_threshold or self.DEFAULT_ALERT_THRESHOLD

        # Load or initialize state
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load drawdown state from file or initialize new state."""
        if self.drawdown_file.exists():
            try:
                with open(self.drawdown_file, 'r') as f:
                    state = json.load(f)
                    # Migrate old state if needed
                    if 'peak_bankroll' not in state:
                        state['peak_bankroll'] = self.starting_bankroll
                    return state
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize new state
        return {
            'peak_bankroll': self.starting_bankroll,
            'current_bankroll': self.starting_bankroll,
            'daily_pnl': {},      # {date_str: total_pnl}
            'weekly_pnl': {},     # {week_str: total_pnl}
            'pnl_history': [],    # List of {time, pnl, position_id, bankroll}
            'halt_reason': None,  # Set if trading is halted
            'halt_time': None,
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        """Save drawdown state to file."""
        self._state['last_updated'] = datetime.now(timezone.utc).isoformat()
        with open(self.drawdown_file, 'w') as f:
            json.dump(self._state, f, indent=2, default=str)

    def _get_date_key(self, dt: datetime = None) -> str:
        """Get date string key for daily tracking."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime('%Y-%m-%d')

    def _get_week_key(self, dt: datetime = None) -> str:
        """Get week string key for weekly tracking (ISO week)."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime('%Y-W%W')

    def sync_bankroll(self, current_bankroll: float):
        """Sync current bankroll from external source (e.g., bankroll.json).

        Call this before checking can_trade() to ensure accurate state.

        Args:
            current_bankroll: Current bankroll value
        """
        self._state['current_bankroll'] = current_bankroll
        # Update peak if new high
        if current_bankroll > self._state['peak_bankroll']:
            self._state['peak_bankroll'] = current_bankroll
        self._save_state()

    def get_daily_pnl(self, date_key: str = None) -> float:
        """Get total P&L for a specific day.

        Args:
            date_key: Date string (YYYY-MM-DD), defaults to today
        """
        if date_key is None:
            date_key = self._get_date_key()
        return self._state['daily_pnl'].get(date_key, 0.0)

    def get_weekly_pnl(self, week_key: str = None) -> float:
        """Get total P&L for a specific week.

        Args:
            week_key: Week string (YYYY-WXX), defaults to current week
        """
        if week_key is None:
            week_key = self._get_week_key()
        return self._state['weekly_pnl'].get(week_key, 0.0)

    def get_total_drawdown(self) -> float:
        """Get current drawdown from peak as a fraction.

        Returns:
            Drawdown as fraction (e.g., -0.15 means 15% below peak)
        """
        peak = self._state['peak_bankroll']
        current = self._state['current_bankroll']
        if peak <= 0:
            return 0.0
        return (current - peak) / peak

    def get_status(self) -> Dict[str, Any]:
        """Get current drawdown status with all metrics.

        Returns:
            Dict with:
                - can_trade: bool
                - halt_reason: str or None
                - daily_pnl: float
                - daily_limit: float
                - daily_pct_used: float
                - weekly_pnl: float
                - weekly_limit: float
                - weekly_pct_used: float
                - total_drawdown: float
                - drawdown_limit: float
                - drawdown_pct_used: float
                - alerts: list of warning messages
        """
        daily_pnl = self.get_daily_pnl()
        weekly_pnl = self.get_weekly_pnl()
        total_drawdown = self.get_total_drawdown()

        daily_limit = self._state['current_bankroll'] * self.max_daily_loss
        weekly_limit = self._state['current_bankroll'] * self.max_weekly_loss
        drawdown_limit = self._state['peak_bankroll'] * self.max_total_drawdown

        # Calculate percentage of limits used (for losses, use absolute values)
        daily_pct = abs(daily_pnl) / daily_limit if daily_limit > 0 else 0
        weekly_pct = abs(weekly_pnl) / weekly_limit if weekly_limit > 0 else 0
        drawdown_pct = abs(total_drawdown) / self.max_total_drawdown if self.max_total_drawdown > 0 else 0

        # Check for alerts
        alerts = []
        if daily_pnl < 0 and daily_pct >= self.alert_threshold:
            alerts.append(f"WARNING: Daily loss at {daily_pct:.0%} of limit")
        if weekly_pnl < 0 and weekly_pct >= self.alert_threshold:
            alerts.append(f"WARNING: Weekly loss at {weekly_pct:.0%} of limit")
        if total_drawdown < 0 and drawdown_pct >= self.alert_threshold:
            alerts.append(f"WARNING: Total drawdown at {drawdown_pct:.0%} of limit")

        can_trade, halt_reason = self._check_limits()

        return {
            'can_trade': can_trade,
            'halt_reason': halt_reason,
            'current_bankroll': self._state['current_bankroll'],
            'peak_bankroll': self._state['peak_bankroll'],
            'daily_pnl': daily_pnl,
            'daily_limit': -daily_limit,  # Negative since it's a loss limit
            'daily_pct_used': daily_pct,
            'weekly_pnl': weekly_pnl,
            'weekly_limit': -weekly_limit,
            'weekly_pct_used': weekly_pct,
            'total_drawdown': total_drawdown,
            'drawdown_limit': -self.max_total_drawdown,
            'drawdown_pct_used': drawdown_pct,
            'alerts': alerts,
        }

    def _check_limits(self) -> tuple:
        """Check if any limits are exceeded.

        Returns:
            (can_trade: bool, halt_reason: str or None)
        """
        # Check if already halted
        if self._state.get('halt_reason'):
            return False, self._state['halt_reason']

        daily_pnl = self.get_daily_pnl()
        weekly_pnl = self.get_weekly_pnl()
        total_drawdown = self.get_total_drawdown()

        current = self._state['current_bankroll']

        # Check daily limit
        daily_limit = current * self.max_daily_loss
        if daily_pnl < 0 and abs(daily_pnl) >= daily_limit:
            return False, f"DAILY_LIMIT: Lost ${abs(daily_pnl):.2f} (>= {self.max_daily_loss:.0%} of bankroll)"

        # Check weekly limit
        weekly_limit = current * self.max_weekly_loss
        if weekly_pnl < 0 and abs(weekly_pnl) >= weekly_limit:
            return False, f"WEEKLY_LIMIT: Lost ${abs(weekly_pnl):.2f} (>= {self.max_weekly_loss:.0%} of bankroll)"

        # Check total drawdown
        if total_drawdown < 0 and abs(total_drawdown) >= self.max_total_drawdown:
            return False, f"TOTAL_DRAWDOWN: {total_drawdown:.1%} from peak (>= {self.max_total_drawdown:.0%} limit)"

        return True, None

    def can_trade(self) -> bool:
        """Check if trading is allowed based on drawdown limits.

        Returns:
            True if trading is allowed, False if any limit is exceeded
        """
        can_trade, _ = self._check_limits()
        return can_trade

    def record_pnl(
        self,
        pnl: float,
        position_id: str = None,
        new_bankroll: float = None
    ) -> Dict[str, Any]:
        """Record P&L from a resolved/closed position.

        Args:
            pnl: Dollar amount won (+) or lost (-)
            position_id: Identifier for the position
            new_bankroll: Updated bankroll after this P&L (optional)

        Returns:
            Dict with updated status including any alerts
        """
        now = datetime.now(timezone.utc)
        date_key = self._get_date_key(now)
        week_key = self._get_week_key(now)

        # Update daily P&L
        if date_key not in self._state['daily_pnl']:
            self._state['daily_pnl'][date_key] = 0.0
        self._state['daily_pnl'][date_key] += pnl

        # Update weekly P&L
        if week_key not in self._state['weekly_pnl']:
            self._state['weekly_pnl'][week_key] = 0.0
        self._state['weekly_pnl'][week_key] += pnl

        # Update bankroll
        if new_bankroll is not None:
            self._state['current_bankroll'] = new_bankroll
            # Update peak if new high
            if new_bankroll > self._state['peak_bankroll']:
                self._state['peak_bankroll'] = new_bankroll
        else:
            self._state['current_bankroll'] += pnl
            if self._state['current_bankroll'] > self._state['peak_bankroll']:
                self._state['peak_bankroll'] = self._state['current_bankroll']

        # Record in history
        self._state['pnl_history'].append({
            'time': now.isoformat(),
            'pnl': pnl,
            'position_id': position_id,
            'bankroll': self._state['current_bankroll'],
            'daily_total': self._state['daily_pnl'][date_key],
            'weekly_total': self._state['weekly_pnl'][week_key],
        })

        # Check if any limits are now exceeded
        can_trade, halt_reason = self._check_limits()
        if not can_trade and not self._state.get('halt_reason'):
            self._state['halt_reason'] = halt_reason
            self._state['halt_time'] = now.isoformat()

        self._save_state()
        return self.get_status()

    def reset_daily(self):
        """Reset daily P&L (for manual reset or testing)."""
        date_key = self._get_date_key()
        self._state['daily_pnl'][date_key] = 0.0
        self._save_state()

    def reset_weekly(self):
        """Reset weekly P&L (for manual reset or testing)."""
        week_key = self._get_week_key()
        self._state['weekly_pnl'][week_key] = 0.0
        self._save_state()

    def reset_halt(self):
        """Clear halt status (for manual override or testing)."""
        self._state['halt_reason'] = None
        self._state['halt_time'] = None
        self._save_state()

    def reset_all(self, new_bankroll: float = None):
        """Reset all state (for testing or starting fresh).

        Args:
            new_bankroll: New starting bankroll (defaults to original starting_bankroll)
        """
        if new_bankroll is None:
            new_bankroll = self.starting_bankroll
        self._state = {
            'peak_bankroll': new_bankroll,
            'current_bankroll': new_bankroll,
            'daily_pnl': {},
            'weekly_pnl': {},
            'pnl_history': [],
            'halt_reason': None,
            'halt_time': None,
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }
        self._save_state()

    def get_history(self, days: int = 7) -> list:
        """Get P&L history for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of P&L records
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        return [
            record for record in self._state['pnl_history']
            if record['time'] >= cutoff_str
        ]


def main():
    """CLI interface for DrawdownManager."""
    import argparse

    parser = argparse.ArgumentParser(description='Drawdown Manager CLI')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--reset-daily', action='store_true', help='Reset daily P&L')
    parser.add_argument('--reset-weekly', action='store_true', help='Reset weekly P&L')
    parser.add_argument('--reset-halt', action='store_true', help='Clear halt status')
    parser.add_argument('--reset-all', action='store_true', help='Reset all state')
    parser.add_argument('--history', type=int, metavar='DAYS', help='Show history for N days')

    args = parser.parse_args()

    dm = DrawdownManager()

    if args.reset_daily:
        dm.reset_daily()
        print("Daily P&L reset")
    elif args.reset_weekly:
        dm.reset_weekly()
        print("Weekly P&L reset")
    elif args.reset_halt:
        dm.reset_halt()
        print("Halt status cleared")
    elif args.reset_all:
        dm.reset_all()
        print("All state reset")
    elif args.history:
        history = dm.get_history(args.history)
        print(f"P&L History (last {args.history} days):")
        for record in history:
            print(f"  {record['time'][:16]}: ${record['pnl']:+.2f} -> ${record['bankroll']:.2f}")
    else:
        # Show status by default
        status = dm.get_status()
        print("=" * 50)
        print("DRAWDOWN MANAGER STATUS")
        print("=" * 50)
        print(f"Can Trade: {'YES' if status['can_trade'] else 'NO'}")
        if status['halt_reason']:
            print(f"Halt Reason: {status['halt_reason']}")
        print()
        print(f"Bankroll: ${status['current_bankroll']:.2f}")
        print(f"Peak:     ${status['peak_bankroll']:.2f}")
        print()
        print(f"Daily P&L:    ${status['daily_pnl']:+.2f} (limit: ${status['daily_limit']:.2f}, {status['daily_pct_used']:.0%} used)")
        print(f"Weekly P&L:   ${status['weekly_pnl']:+.2f} (limit: ${status['weekly_limit']:.2f}, {status['weekly_pct_used']:.0%} used)")
        print(f"Total DD:     {status['total_drawdown']:.1%} (limit: {status['drawdown_limit']:.0%}, {status['drawdown_pct_used']:.0%} used)")

        if status['alerts']:
            print()
            for alert in status['alerts']:
                print(f"  {alert}")


if __name__ == "__main__":
    main()
