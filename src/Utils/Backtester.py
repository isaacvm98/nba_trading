"""
Backtester for NBA Betting System

Simulates trading with different strategies using historical position data
to compare old system (with early exits) vs new system (no early exits).

Usage:
    python -m src.Utils.Backtester --compare
    python -m src.Utils.Backtester --simulate-new
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class Backtester:
    """Simulates trading strategies using historical data."""

    def __init__(self, data_dir: Path = Path("Data/paper_trading")):
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

    def get_all_positions_with_outcomes(self) -> List[Dict[str, Any]]:
        """Get all positions that have final outcomes (resolved or closed).

        Returns list of positions with:
        - Original entry data
        - Exit type (resolved vs early_exit)
        - Actual P&L
        - What P&L would have been if ran to resolution
        """
        positions = self._load_positions()
        trades = self._load_trades()

        # Build trade lookup by position_id
        trade_lookup = defaultdict(list)
        for trade in trades:
            pos_id = trade.get('position_id')
            if pos_id:
                trade_lookup[pos_id].append(trade)

        results = []
        for pos_id, pos in positions.items():
            if not pos.get('bet_side'):
                continue

            status = pos.get('status')
            if status not in ['resolved', 'closed']:
                continue

            result = {
                'position_id': pos_id,
                'game': f"{pos.get('away_team')} @ {pos.get('home_team')}",
                'bet_side': pos.get('bet_side'),
                'bet_kelly': pos.get('bet_kelly', 0),
                'entry_edge': pos.get(f"{pos.get('bet_side')}_edge", 0),
                'status': status,
                'actual_pnl': pos.get('pnl', 0),
                'exit_reason': pos.get('exit_reason', 'unknown'),
            }

            # For closed positions (early exits), try to find the resolution
            if status == 'closed':
                result['exit_type'] = 'early_exit'
                # Look for a RESOLVED trade for the same game
                for trade in trades:
                    if (trade.get('type') == 'RESOLVED' and
                        trade.get('game') == result['game']):
                        # Found the resolution
                        result['would_have_won'] = trade.get('won', False)
                        result['resolution_pnl'] = self._estimate_resolution_pnl(
                            pos, trade.get('won', False)
                        )
                        break
            else:
                result['exit_type'] = 'resolved'
                result['would_have_won'] = pos.get('won')
                result['resolution_pnl'] = pos.get('pnl', 0)

            results.append(result)

        return results

    def _estimate_resolution_pnl(self, position: Dict, won: bool) -> float:
        """Estimate what P&L would have been if position ran to resolution.

        Args:
            position: Position data dict
            won: Whether the bet would have won

        Returns:
            Estimated P&L in dollars
        """
        bet_kelly = position.get('bet_kelly', 0)
        # Assuming $1000 bankroll and kelly is in %
        stake = 1000 * (bet_kelly / 100)

        if won:
            # Calculate potential payout based on odds
            bet_side = position.get('bet_side')
            if bet_side == 'home':
                entry_prob = position.get('entry_home_prob', 0.5)
            else:
                entry_prob = position.get('entry_away_prob', 0.5)

            # Payout = stake * (1/prob - 1) for a winning bet
            if entry_prob > 0:
                payout = stake * (1 / entry_prob - 1)
            else:
                payout = stake
            return payout
        else:
            return -stake

    def simulate_strategy(
        self,
        positions: List[Dict[str, Any]],
        use_early_exits: bool = False
    ) -> Dict[str, Any]:
        """Simulate a trading strategy.

        Args:
            positions: List of position results
            use_early_exits: If True, use actual P&L. If False, use resolution P&L.

        Returns:
            Dict with strategy results
        """
        total_pnl = 0.0
        wins = 0
        losses = 0
        total_bets = 0

        for pos in positions:
            if not pos.get('bet_kelly'):
                continue

            total_bets += 1

            if use_early_exits:
                pnl = pos.get('actual_pnl', 0)
            else:
                pnl = pos.get('resolution_pnl', pos.get('actual_pnl', 0))

            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        return {
            'strategy': 'With Early Exits' if use_early_exits else 'Run to Resolution',
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl_per_bet': round(total_pnl / total_bets, 2) if total_bets > 0 else 0,
        }

    def compare_strategies(self) -> Dict[str, Any]:
        """Compare old system (with early exits) vs new system (no early exits).

        Returns:
            Comparison results
        """
        positions = self.get_all_positions_with_outcomes()

        # Only include positions where we have both outcomes
        complete_positions = [
            p for p in positions
            if 'resolution_pnl' in p or p.get('exit_type') == 'resolved'
        ]

        old_system = self.simulate_strategy(complete_positions, use_early_exits=True)
        new_system = self.simulate_strategy(complete_positions, use_early_exits=False)

        # Calculate difference
        pnl_diff = new_system['total_pnl'] - old_system['total_pnl']

        return {
            'positions_analyzed': len(complete_positions),
            'old_system': old_system,
            'new_system': new_system,
            'pnl_improvement': round(pnl_diff, 2),
            'percentage_improvement': round(pnl_diff / abs(old_system['total_pnl']) * 100, 1)
                if old_system['total_pnl'] != 0 else 0,
        }

    def analyze_early_exit_impact(self) -> Dict[str, Any]:
        """Analyze the impact of early exits on P&L.

        Returns:
            Analysis of how early exits affected outcomes
        """
        positions = self.get_all_positions_with_outcomes()

        early_exits = [p for p in positions if p.get('exit_type') == 'early_exit']
        resolved = [p for p in positions if p.get('exit_type') == 'resolved']

        # For early exits, see how many would have won
        would_have_won = sum(1 for p in early_exits if p.get('would_have_won'))
        total_early_exits = len(early_exits)

        # Calculate lost profit from early exits
        lost_profit = sum(
            (p.get('resolution_pnl', 0) - p.get('actual_pnl', 0))
            for p in early_exits
            if p.get('would_have_won')
        )

        return {
            'total_early_exits': total_early_exits,
            'would_have_won': would_have_won,
            'would_have_won_pct': round(would_have_won / total_early_exits * 100, 1)
                if total_early_exits > 0 else 0,
            'actual_early_exit_pnl': round(sum(p.get('actual_pnl', 0) for p in early_exits), 2),
            'potential_resolution_pnl': round(sum(p.get('resolution_pnl', 0) for p in early_exits), 2),
            'lost_profit': round(lost_profit, 2),
            'resolved_positions': len(resolved),
            'resolved_win_rate': round(sum(1 for p in resolved if p.get('actual_pnl', 0) > 0)
                / len(resolved) * 100, 1) if resolved else 0,
            'resolved_pnl': round(sum(p.get('actual_pnl', 0) for p in resolved), 2),
        }

    def analyze_underdog_take_profit(
        self,
        buckets=None,
        thresholds=None,
        bankroll=1000.0,
    ) -> Dict[str, Any]:
        """Analyze optimal take-profit thresholds for underdog positions.

        Groups resolved underdog positions by entry probability bucket,
        then simulates different take-profit thresholds to find the optimal
        one for each bucket.

        Args:
            buckets: List of (name, min_prob, max_prob) tuples.
            thresholds: List of take-profit percentages to test.
            bankroll: Assumed bankroll for P&L calculations.

        Returns:
            Dict with per-bucket analysis including optimal thresholds.
        """
        if buckets is None:
            buckets = [
                ('0-20%', 0, 0.20),
                ('20-35%', 0.20, 0.35),
                ('35-50%', 0.35, 0.50),
            ]

        if thresholds is None:
            thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50, 1.00]

        positions = self._load_positions()

        # Filter to resolved underdog positions with bets
        underdog_positions = []
        for pos_id, pos in positions.items():
            bet_side = pos.get('bet_side')
            if not bet_side:
                continue
            if pos.get('status') != 'resolved':
                continue

            entry_prob = pos.get('entry_home_prob') if bet_side == 'home' else pos.get('entry_away_prob')
            if entry_prob is None or entry_prob >= 0.50:
                continue

            underdog_positions.append({
                'position_id': pos_id,
                'entry_prob': entry_prob,
                'max_profit_pct': pos.get('max_profit_pct'),
                'won': pos.get('won', False),
                'pnl': pos.get('pnl', 0),
                'stake': bankroll * (pos.get('bet_kelly', 0) / 100),
            })

        results = {
            'total_underdog_resolved': len(underdog_positions),
            'with_tracking': sum(1 for p in underdog_positions if p['max_profit_pct'] is not None),
            'buckets': {},
        }

        for bucket_name, min_prob, max_prob in buckets:
            bucket_positions = [
                p for p in underdog_positions
                if min_prob <= p['entry_prob'] < max_prob
            ]

            if not bucket_positions:
                results['buckets'][bucket_name] = {'count': 0, 'recommendation': 'insufficient_data'}
                continue

            wins = sum(1 for p in bucket_positions if p['won'])
            losses = len(bucket_positions) - wins
            baseline_pnl = sum(p['pnl'] for p in bucket_positions)
            with_tracking = sum(1 for p in bucket_positions if p['max_profit_pct'] is not None)

            avg_entry = sum(p['entry_prob'] for p in bucket_positions) / len(bucket_positions)
            win_payout_multiplier = (1 / avg_entry) - 1 if avg_entry > 0 else 1

            # Simulate each threshold
            threshold_results = []
            for t in thresholds:
                sim_pnl = 0
                exits_triggered = 0

                for p in bucket_positions:
                    if p['max_profit_pct'] is not None and p['max_profit_pct'] >= t:
                        # Would have exited at take-profit
                        sim_pnl += p['stake'] * t
                        exits_triggered += 1
                    else:
                        # No TP trigger or no tracking data -> use actual resolution P&L
                        sim_pnl += p['pnl']

                improvement = sim_pnl - baseline_pnl
                threshold_results.append({
                    'threshold': t,
                    'sim_pnl': round(sim_pnl, 2),
                    'improvement': round(improvement, 2),
                    'exits_triggered': exits_triggered,
                    'exit_rate': round(exits_triggered / len(bucket_positions), 3),
                })

            best = max(threshold_results, key=lambda x: x['improvement'])

            if best['improvement'] <= 0:
                recommendation = 'hold_to_resolution'
                optimal_threshold = None
            elif with_tracking < 5:
                recommendation = 'insufficient_tracking_data'
                optimal_threshold = best['threshold']
            else:
                recommendation = 'use_take_profit'
                optimal_threshold = best['threshold']

            results['buckets'][bucket_name] = {
                'count': len(bucket_positions),
                'wins': wins,
                'losses': losses,
                'win_rate': round(wins / len(bucket_positions), 3),
                'with_tracking': with_tracking,
                'baseline_pnl': round(baseline_pnl, 2),
                'avg_entry_prob': round(avg_entry, 3),
                'threshold_results': threshold_results,
                'optimal_threshold': optimal_threshold,
                'best_improvement': round(best['improvement'], 2),
                'recommendation': recommendation,
            }

        return results

    def generate_underdog_tp_report(self) -> str:
        """Generate a report on underdog take-profit optimization."""
        analysis = self.analyze_underdog_take_profit()

        lines = []
        lines.append("=" * 70)
        lines.append("         UNDERDOG TAKE-PROFIT OPTIMIZATION REPORT")
        lines.append(f"         {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        lines.append(f"\nTotal resolved underdogs: {analysis['total_underdog_resolved']}")
        lines.append(f"With price tracking data: {analysis['with_tracking']}")

        for bucket_name, bucket_data in analysis['buckets'].items():
            lines.append(f"\n--- Bucket: {bucket_name} ---")
            if bucket_data['count'] == 0:
                lines.append("  No positions in this bucket")
                continue

            lines.append(f"  Positions: {bucket_data['count']} ({bucket_data['wins']}W / {bucket_data['losses']}L)")
            lines.append(f"  Win Rate: {bucket_data['win_rate']*100:.1f}%")
            lines.append(f"  Avg Entry Prob: {bucket_data['avg_entry_prob']*100:.1f}%")
            lines.append(f"  Baseline P&L (hold): ${bucket_data['baseline_pnl']:+.2f}")
            lines.append(f"  Tracking data: {bucket_data['with_tracking']}/{bucket_data['count']}")

            lines.append(f"\n  Recommendation: {bucket_data['recommendation'].upper()}")
            if bucket_data['optimal_threshold']:
                lines.append(f"  Optimal threshold: {bucket_data['optimal_threshold']*100:.0f}%")
                lines.append(f"  Improvement: ${bucket_data['best_improvement']:+.2f}")

            lines.append(f"\n  Threshold Simulation:")
            for tr in bucket_data.get('threshold_results', []):
                lines.append(
                    f"    TP@{tr['threshold']*100:5.1f}%: "
                    f"P&L=${tr['sim_pnl']:+8.2f} "
                    f"(diff: ${tr['improvement']:+8.2f}) "
                    f"| {tr['exits_triggered']}/{bucket_data['count']} exited"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def generate_report(self) -> str:
        """Generate a comprehensive backtest report."""
        lines = []
        lines.append("=" * 70)
        lines.append("                    BACKTEST SIMULATION REPORT")
        lines.append(f"                    {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)

        # Strategy Comparison
        comparison = self.compare_strategies()
        lines.append("\n--- STRATEGY COMPARISON ---")
        lines.append(f"Positions analyzed: {comparison['positions_analyzed']}")

        old = comparison['old_system']
        new = comparison['new_system']

        lines.append(f"\nOLD SYSTEM (With Early Exits):")
        lines.append(f"  Bets: {old['total_bets']} | Wins: {old['wins']} | Win Rate: {old['win_rate']*100:.1f}%")
        lines.append(f"  Total P&L: ${old['total_pnl']:+.2f} | Avg/Bet: ${old['avg_pnl_per_bet']:+.2f}")

        lines.append(f"\nNEW SYSTEM (Run to Resolution):")
        lines.append(f"  Bets: {new['total_bets']} | Wins: {new['wins']} | Win Rate: {new['win_rate']*100:.1f}%")
        lines.append(f"  Total P&L: ${new['total_pnl']:+.2f} | Avg/Bet: ${new['avg_pnl_per_bet']:+.2f}")

        lines.append(f"\nIMPROVEMENT:")
        lines.append(f"  P&L Difference: ${comparison['pnl_improvement']:+.2f}")
        if comparison['percentage_improvement'] != 0:
            lines.append(f"  Percentage: {comparison['percentage_improvement']:+.1f}%")

        # Early Exit Impact Analysis
        impact = self.analyze_early_exit_impact()
        lines.append("\n--- EARLY EXIT IMPACT ---")
        lines.append(f"Total early exits: {impact['total_early_exits']}")
        lines.append(f"Would have won: {impact['would_have_won']} ({impact['would_have_won_pct']:.1f}%)")
        lines.append(f"Actual early exit P&L: ${impact['actual_early_exit_pnl']:+.2f}")
        lines.append(f"If ran to resolution: ${impact['potential_resolution_pnl']:+.2f}")
        lines.append(f"Lost profit from early exits: ${impact['lost_profit']:+.2f}")

        lines.append(f"\nResolved positions: {impact['resolved_positions']}")
        lines.append(f"Resolved win rate: {impact['resolved_win_rate']:.1f}%")
        lines.append(f"Resolved P&L: ${impact['resolved_pnl']:+.2f}")

        # Recommendations
        lines.append("\n--- RECOMMENDATIONS ---")
        if comparison['pnl_improvement'] > 0:
            lines.append("* New system (no early exits) outperforms old system")
            lines.append(f"* Estimated profit improvement: ${comparison['pnl_improvement']:+.2f}")
            lines.append("* Recommendation: Keep early exits DISABLED")
        else:
            lines.append("* Old system (with early exits) outperforms new system")
            lines.append("* Consider re-enabling early exits or adjusting thresholds")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Simulation')
    parser.add_argument('--compare', action='store_true', help='Compare old vs new strategy')
    parser.add_argument('--impact', action='store_true', help='Analyze early exit impact')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--underdog-tp', action='store_true', help='Analyze underdog take-profit optimization')

    args = parser.parse_args()

    backtester = Backtester()

    if args.compare:
        comparison = backtester.compare_strategies()
        print(json.dumps(comparison, indent=2))
    elif args.impact:
        impact = backtester.analyze_early_exit_impact()
        print(json.dumps(impact, indent=2))
    elif args.underdog_tp:
        report = backtester.generate_underdog_tp_report()
        print(report)
    else:
        # Default: generate full report
        report = backtester.generate_report()
        print(report)


if __name__ == "__main__":
    main()
