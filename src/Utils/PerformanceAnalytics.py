"""
Performance Analytics for NBA Betting System

Provides weekly reporting and model validation:
- Win rate by edge bucket (5-7%, 7-10%, 10%+)
- P&L by edge bucket
- Model calibration check (predicted prob vs actual win rate)
- Alerts for miscalibration

Usage:
    python -m src.Utils.PerformanceAnalytics --report
    python -m src.Utils.PerformanceAnalytics --calibration
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional


class PerformanceAnalytics:
    """Analyzes trading performance and model calibration."""

    # Edge buckets for analysis
    EDGE_BUCKETS = [
        ("5-7%", 0.05, 0.07),
        ("7-10%", 0.07, 0.10),
        ("10%+", 0.10, 1.00),
    ]

    # Calibration thresholds
    MIN_WIN_RATE = 0.52  # Below this is concerning
    MAX_WIN_RATE = 0.70  # Above this is suspicious (likely overfitting or data issue)

    def __init__(self, data_dir: Path = Path("Data/paper_trading")):
        """Initialize PerformanceAnalytics.

        Args:
            data_dir: Directory containing positions.json and trades.json
        """
        self.data_dir = Path(data_dir)
        self.positions_file = self.data_dir / "positions.json"
        self.trades_file = self.data_dir / "trades.json"

    def _load_positions(self) -> Dict[str, Any]:
        """Load positions from file."""
        if self.positions_file.exists():
            with open(self.positions_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trades log from file."""
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return []

    def get_resolved_positions(self, days: int = None) -> List[Dict[str, Any]]:
        """Get resolved positions, optionally filtered by days.

        Args:
            days: Only include positions from the last N days (None = all)

        Returns:
            List of resolved position dicts with bet_side
        """
        positions = self._load_positions()
        resolved = []

        cutoff = None
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_str = cutoff.isoformat()

        for pos_id, pos in positions.items():
            # Only include resolved positions with actual bets
            if pos.get('status') != 'resolved':
                continue
            if not pos.get('bet_side'):
                continue

            # Check date filter
            if cutoff and pos.get('exit_time', '') < cutoff_str:
                continue

            resolved.append(pos)

        return resolved

    def get_closed_positions(self, days: int = None) -> List[Dict[str, Any]]:
        """Get early-exit (closed) positions, optionally filtered by days."""
        positions = self._load_positions()
        closed = []

        cutoff = None
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_str = cutoff.isoformat()

        for pos_id, pos in positions.items():
            if pos.get('status') != 'closed':
                continue
            if not pos.get('bet_side'):
                continue

            if cutoff and pos.get('exit_time', '') < cutoff_str:
                continue

            closed.append(pos)

        return closed

    def _get_edge_bucket(self, edge: float) -> Optional[str]:
        """Get the bucket name for a given edge."""
        for name, low, high in self.EDGE_BUCKETS:
            if low <= edge < high:
                return name
        return None

    def analyze_by_edge_bucket(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Analyze performance by edge bucket.

        Returns:
            Dict with bucket names as keys, containing:
                - count: number of bets
                - wins: number of wins
                - win_rate: win percentage
                - total_pnl: total P&L
                - avg_edge: average edge
        """
        buckets = defaultdict(lambda: {
            'count': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'edges': []
        })

        for pos in positions:
            bet_side = pos.get('bet_side')
            if not bet_side:
                continue

            edge = abs(pos.get(f'{bet_side}_edge', 0))
            bucket = self._get_edge_bucket(edge)
            if not bucket:
                continue

            buckets[bucket]['count'] += 1
            buckets[bucket]['edges'].append(edge)
            buckets[bucket]['total_pnl'] += pos.get('pnl', 0)

            if pos.get('won'):
                buckets[bucket]['wins'] += 1

        # Calculate derived stats
        result = {}
        for bucket_name, data in buckets.items():
            count = data['count']
            result[bucket_name] = {
                'count': count,
                'wins': data['wins'],
                'win_rate': data['wins'] / count if count > 0 else 0,
                'total_pnl': round(data['total_pnl'], 2),
                'avg_edge': sum(data['edges']) / len(data['edges']) if data['edges'] else 0,
            }

        return result

    def analyze_calibration(self, positions: List[Dict[str, Any]], num_bins: int = 5) -> Dict[str, Any]:
        """Analyze model calibration (predicted prob vs actual win rate).

        Bins positions by predicted probability and compares to actual win rate.

        Returns:
            Dict containing:
                - bins: list of {predicted_prob, actual_win_rate, count}
                - is_calibrated: bool (True if calibration looks good)
                - calibration_error: mean absolute error between predicted and actual
                - alerts: list of warning messages
        """
        # Group by predicted probability bins
        bin_size = 1.0 / num_bins
        bins = defaultdict(lambda: {'predictions': [], 'actuals': []})

        for pos in positions:
            bet_side = pos.get('bet_side')
            if not bet_side:
                continue

            # Get predicted probability for the side we bet
            if bet_side == 'home':
                pred_prob = pos.get('adjusted_home_prob') or pos.get('model_home_prob', 0.5)
            else:
                pred_prob = pos.get('adjusted_away_prob') or pos.get('model_away_prob', 0.5)

            # Bin by predicted probability
            bin_idx = min(int(pred_prob / bin_size), num_bins - 1)
            bin_center = (bin_idx + 0.5) * bin_size

            bins[bin_idx]['predictions'].append(pred_prob)
            bins[bin_idx]['actuals'].append(1 if pos.get('won') else 0)

        # Calculate calibration
        result_bins = []
        total_error = 0.0
        total_count = 0

        for bin_idx in sorted(bins.keys()):
            data = bins[bin_idx]
            count = len(data['predictions'])
            if count == 0:
                continue

            predicted_avg = sum(data['predictions']) / count
            actual_avg = sum(data['actuals']) / count

            result_bins.append({
                'bin': f"{bin_idx * bin_size:.0%}-{(bin_idx + 1) * bin_size:.0%}",
                'predicted_prob': round(predicted_avg, 3),
                'actual_win_rate': round(actual_avg, 3),
                'count': count,
                'error': round(abs(predicted_avg - actual_avg), 3)
            })

            total_error += abs(predicted_avg - actual_avg) * count
            total_count += count

        calibration_error = total_error / total_count if total_count > 0 else 0

        # Check for calibration issues
        alerts = []
        overall_win_rate = sum(1 for p in positions if p.get('won')) / len(positions) if positions else 0

        if overall_win_rate < self.MIN_WIN_RATE:
            alerts.append(f"LOW WIN RATE: {overall_win_rate:.1%} (< {self.MIN_WIN_RATE:.0%})")
        if overall_win_rate > self.MAX_WIN_RATE:
            alerts.append(f"SUSPICIOUSLY HIGH WIN RATE: {overall_win_rate:.1%} (> {self.MAX_WIN_RATE:.0%})")
        if calibration_error > 0.10:
            alerts.append(f"POOR CALIBRATION: Mean error {calibration_error:.1%}")

        return {
            'bins': result_bins,
            'overall_win_rate': round(overall_win_rate, 3),
            'calibration_error': round(calibration_error, 3),
            'is_calibrated': calibration_error < 0.10 and len(alerts) == 0,
            'alerts': alerts,
            'total_positions': len(positions),
        }

    def generate_weekly_report(self, days: int = 7) -> str:
        """Generate a weekly performance report.

        Args:
            days: Number of days to include (default 7)

        Returns:
            Formatted report string
        """
        resolved = self.get_resolved_positions(days)
        closed = self.get_closed_positions(days)

        lines = []
        lines.append("=" * 60)
        lines.append(f"PERFORMANCE REPORT - Last {days} Days")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        # Overall stats
        lines.append("\n--- OVERALL STATS ---")
        lines.append(f"Resolved positions: {len(resolved)}")
        lines.append(f"Early exits: {len(closed)}")

        total_pnl = sum(p.get('pnl', 0) for p in resolved + closed)
        lines.append(f"Total P&L: ${total_pnl:+.2f}")

        if resolved:
            resolved_wins = sum(1 for p in resolved if p.get('won'))
            resolved_pnl = sum(p.get('pnl', 0) for p in resolved)
            lines.append(f"\nResolved: {resolved_wins}/{len(resolved)} wins ({resolved_wins/len(resolved)*100:.1f}%)")
            lines.append(f"Resolved P&L: ${resolved_pnl:+.2f}")

        if closed:
            closed_wins = sum(1 for p in closed if p.get('pnl', 0) > 0)
            closed_pnl = sum(p.get('pnl', 0) for p in closed)
            lines.append(f"\nEarly exits: {closed_wins}/{len(closed)} profitable ({closed_wins/len(closed)*100:.1f}%)")
            lines.append(f"Early exit P&L: ${closed_pnl:+.2f}")

        # Edge bucket analysis (resolved only)
        if resolved:
            lines.append("\n--- WIN RATE BY EDGE BUCKET ---")
            bucket_stats = self.analyze_by_edge_bucket(resolved)
            for bucket_name, _, _ in self.EDGE_BUCKETS:
                if bucket_name in bucket_stats:
                    stats = bucket_stats[bucket_name]
                    lines.append(f"  {bucket_name}: {stats['wins']}/{stats['count']} ({stats['win_rate']*100:.1f}%) | P&L: ${stats['total_pnl']:+.2f}")
                else:
                    lines.append(f"  {bucket_name}: No data")

        # Calibration check (resolved only)
        if resolved:
            lines.append("\n--- MODEL CALIBRATION ---")
            calibration = self.analyze_calibration(resolved)
            lines.append(f"Overall win rate: {calibration['overall_win_rate']*100:.1f}%")
            lines.append(f"Calibration error: {calibration['calibration_error']*100:.1f}%")
            lines.append(f"Calibrated: {'YES' if calibration['is_calibrated'] else 'NO'}")

            if calibration['alerts']:
                lines.append("\nAlerts:")
                for alert in calibration['alerts']:
                    lines.append(f"  * {alert}")

            lines.append("\nPredicted vs Actual by Probability Bin:")
            for bin_data in calibration['bins']:
                lines.append(f"  {bin_data['bin']}: pred={bin_data['predicted_prob']*100:.0f}% actual={bin_data['actual_win_rate']*100:.0f}% (n={bin_data['count']})")

        # Max drawdown / profit stats
        if resolved:
            lines.append("\n--- POSITION VARIANCE ---")
            max_drawdowns = [p.get('max_drawdown_pct', 0) for p in resolved if p.get('max_drawdown_pct') is not None]
            max_profits = [p.get('max_profit_pct', 0) for p in resolved if p.get('max_profit_pct') is not None]

            if max_drawdowns:
                avg_drawdown = sum(max_drawdowns) / len(max_drawdowns)
                worst_drawdown = min(max_drawdowns)
                lines.append(f"Avg max drawdown during position: {avg_drawdown*100:.1f}%")
                lines.append(f"Worst drawdown during position: {worst_drawdown*100:.1f}%")

            if max_profits:
                avg_profit = sum(max_profits) / len(max_profits)
                best_profit = max(max_profits)
                lines.append(f"Avg max profit during position: {avg_profit*100:.1f}%")
                lines.append(f"Best profit during position: {best_profit*100:.1f}%")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def save_report(self, report: str, filename: str = None):
        """Save report to file.

        Args:
            report: Report string to save
            filename: Output filename (defaults to report_YYYY-MM-DD.txt)
        """
        if filename is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f"weekly_report_{date_str}.txt"

        output_path = self.data_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Performance Analytics')
    parser.add_argument('--report', action='store_true', help='Generate weekly report')
    parser.add_argument('--calibration', action='store_true', help='Show calibration analysis')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--save', action='store_true', help='Save report to file')

    args = parser.parse_args()

    analytics = PerformanceAnalytics()

    if args.calibration:
        resolved = analytics.get_resolved_positions(args.days)
        if not resolved:
            print("No resolved positions found.")
            return

        calibration = analytics.analyze_calibration(resolved)
        print("=" * 50)
        print(f"CALIBRATION ANALYSIS (last {args.days} days)")
        print("=" * 50)
        print(f"Total positions: {calibration['total_positions']}")
        print(f"Overall win rate: {calibration['overall_win_rate']*100:.1f}%")
        print(f"Calibration error: {calibration['calibration_error']*100:.1f}%")
        print(f"Is calibrated: {'YES' if calibration['is_calibrated'] else 'NO'}")

        if calibration['alerts']:
            print("\nAlerts:")
            for alert in calibration['alerts']:
                print(f"  * {alert}")

        print("\nPredicted vs Actual by Probability Bin:")
        for bin_data in calibration['bins']:
            print(f"  {bin_data['bin']}: pred={bin_data['predicted_prob']*100:.0f}% actual={bin_data['actual_win_rate']*100:.0f}% (n={bin_data['count']})")

    else:
        # Default: generate report
        report = analytics.generate_weekly_report(args.days)
        print(report)

        if args.save:
            analytics.save_report(report)


if __name__ == "__main__":
    main()
