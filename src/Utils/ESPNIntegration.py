"""
ESPN Integration Utilities

Helper functions to integrate ESPN data with the existing betting system.
Provides win probability signals and injury data for enhanced decision making.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.DataProviders.ESPNProvider import ESPNProvider


class ESPNSignalGenerator:
    """
    Generates trading signals from ESPN win probability data.

    Can be used to:
    - Compare model predictions vs ESPN's live win probability
    - Identify games where market/ESPN probability diverges from model
    - Adjust positions during live games based on probability changes
    """

    def __init__(self):
        """Initialize the signal generator with ESPN provider."""
        self.provider = ESPNProvider()

    def get_pregame_signals(self, model_predictions: Dict[str, float]) -> Dict[str, Dict]:
        """
        Compare model predictions against ESPN's pregame win probabilities.

        Args:
            model_predictions: Dict of {team_name: model_win_probability}
                               e.g., {"Los Angeles Lakers": 0.65}

        Returns:
            Dict with signal data for each game:
            {
                "home:away": {
                    "model_home_prob": float,
                    "espn_home_prob": float,
                    "divergence": float,  # model - espn
                    "signal": str,  # "strong_buy", "buy", "hold", "sell", "strong_sell"
                    "confidence": float  # 0-1 based on divergence magnitude
                }
            }
        """
        signals = {}
        espn_probs = self.provider.get_all_live_win_probabilities()

        for game_key, espn_data in espn_probs.items():
            home_team = espn_data['home_team']
            away_team = espn_data['away_team']
            espn_home_prob = espn_data.get('home_team_prob')

            # Skip if no ESPN probability available
            if espn_home_prob is None:
                continue

            # Look for model prediction
            model_home_prob = model_predictions.get(home_team)
            model_away_prob = model_predictions.get(away_team)

            # If we have model prediction for away team, convert to home probability
            if model_home_prob is None and model_away_prob is not None:
                model_home_prob = 1 - model_away_prob

            if model_home_prob is None:
                continue

            # Calculate divergence (positive = model more bullish on home)
            divergence = model_home_prob - espn_home_prob

            # Generate signal
            signal, confidence = self._calculate_signal(divergence)

            signals[game_key] = {
                'home_team': home_team,
                'away_team': away_team,
                'model_home_prob': round(model_home_prob, 3),
                'model_away_prob': round(1 - model_home_prob, 3),
                'espn_home_prob': round(espn_home_prob, 3),
                'espn_away_prob': round(1 - espn_home_prob, 3),
                'divergence': round(divergence, 3),
                'signal': signal,
                'confidence': round(confidence, 3),
                'game_status': espn_data.get('status_detail', ''),
                'is_live': espn_data.get('is_live', False)
            }

        return signals

    def _calculate_signal(self, divergence: float) -> Tuple[str, float]:
        """
        Calculate trading signal based on divergence.

        Args:
            divergence: Model probability - ESPN probability

        Returns:
            Tuple of (signal_name, confidence)
        """
        abs_div = abs(divergence)

        if abs_div < 0.03:
            return ('hold', abs_div / 0.03)
        elif abs_div < 0.07:
            signal = 'buy_home' if divergence > 0 else 'buy_away'
            confidence = (abs_div - 0.03) / 0.04
            return (signal, confidence)
        elif abs_div < 0.12:
            signal = 'strong_buy_home' if divergence > 0 else 'strong_buy_away'
            confidence = min((abs_div - 0.07) / 0.05, 1.0)
            return (signal, confidence)
        else:
            # Very high divergence - might indicate stale data or special circumstances
            signal = 'extreme_home' if divergence > 0 else 'extreme_away'
            return (signal, 1.0)

    def get_live_probability_change(self, event_id: str) -> Optional[Dict]:
        """
        Get probability change trajectory for a live game.

        Useful for identifying momentum shifts during games.

        Args:
            event_id: ESPN event ID

        Returns:
            Dict with probability change data
        """
        prob_data = self.provider.get_win_probability(event_id)
        if not prob_data:
            return None

        items = prob_data.get('items', [])
        if len(items) < 2:
            return None

        # Get probability changes
        probs = [item.get('homeWinPercentage', 0.5) for item in items]

        # Calculate recent momentum (last 5 updates)
        recent = probs[-5:] if len(probs) >= 5 else probs
        momentum = recent[-1] - recent[0]

        # Calculate overall change
        total_change = probs[-1] - probs[0]

        return {
            'current_home_prob': probs[-1],
            'initial_home_prob': probs[0],
            'total_change': round(total_change, 3),
            'recent_momentum': round(momentum, 3),
            'data_points': len(probs),
            'momentum_direction': 'home_gaining' if momentum > 0.02 else
                                  'away_gaining' if momentum < -0.02 else 'stable'
        }


class ESPNInjuryEnhancer:
    """
    Enhances injury data using ESPN's injury reports.

    Can be used alongside existing InjuryProvider for:
    - Cross-validation of injury status
    - Additional injury details (return dates, specific injuries)
    - Historical injury tracking
    """

    def __init__(self):
        """Initialize with ESPN provider."""
        self.provider = ESPNProvider()

    def get_enhanced_injuries(self, team_name: str) -> Dict:
        """
        Get enhanced injury data for a team.

        Args:
            team_name: Full team name

        Returns:
            Dict with injury details
        """
        injuries = self.provider.get_team_injuries(team_name)

        if not injuries:
            return {
                'team': team_name,
                'injuries': [],
                'total_out': 0,
                'source': 'espn'
            }

        parsed_injuries = []
        total_out = 0

        for injury in injuries:
            details = injury.get('details', {})
            injury_type = injury.get('type', {})

            status = injury_type.get('description', injury.get('status', 'unknown'))

            parsed = {
                'status': status,
                'injury_type': details.get('type', 'Unknown'),
                'location': details.get('location', ''),
                'detail': details.get('detail', ''),
                'side': details.get('side', ''),
                'return_date': details.get('returnDate'),
                'short_comment': injury.get('shortComment', ''),
                'date_updated': injury.get('date')
            }

            # Try to get athlete name from the ref URL
            athlete_ref = injury.get('athlete', {}).get('$ref', '')
            if athlete_ref:
                parsed['athlete_ref'] = athlete_ref

            parsed_injuries.append(parsed)

            if status.lower() == 'out':
                total_out += 1

        return {
            'team': team_name,
            'injuries': parsed_injuries,
            'total_out': total_out,
            'total_injured': len(parsed_injuries),
            'source': 'espn'
        }

    def compare_with_tank01(self, team_name: str, tank01_injuries: Dict) -> Dict:
        """
        Compare ESPN injuries with Tank01 data.

        Useful for cross-validation and identifying discrepancies.

        Args:
            team_name: Full team name
            tank01_injuries: Injury data from InjuryProvider

        Returns:
            Comparison dict
        """
        espn_data = self.get_enhanced_injuries(team_name)

        return {
            'team': team_name,
            'espn_total_out': espn_data['total_out'],
            'tank01_total_impact': tank01_injuries.get('total_impact', 0),
            'tank01_injured_count': len(tank01_injuries.get('injured_players', [])),
            'espn_injured_count': espn_data['total_injured'],
            'espn_injuries': espn_data['injuries'],
            'tank01_injuries': tank01_injuries.get('injured_players', [])
        }


def probability_to_odds(prob: float) -> int:
    """
    Convert probability to American odds.

    Args:
        prob: Win probability (0-1)

    Returns:
        American odds (positive for underdogs, negative for favorites)
    """
    if prob <= 0 or prob >= 1:
        return 0

    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))


def odds_to_probability(odds: int) -> float:
    """
    Convert American odds to probability.

    Args:
        odds: American odds

    Returns:
        Win probability (0-1)
    """
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)


if __name__ == "__main__":
    # Demo the ESPN integration
    print("=" * 60)
    print("ESPN Integration Demo")
    print("=" * 60)

    # Test signal generator
    print("\n1. Win Probability Signals:")
    print("-" * 40)

    signal_gen = ESPNSignalGenerator()

    # Get current games and their probabilities
    probs = signal_gen.provider.get_all_live_win_probabilities()

    if probs:
        for game_key, data in probs.items():
            print(f"\n{data['away_team']} @ {data['home_team']}")
            if data['home_team_prob']:
                print(f"  ESPN Win Probability:")
                print(f"    {data['home_team']}: {data['home_team_prob']:.1%}")
                print(f"    {data['away_team']}: {data['away_team_prob']:.1%}")
                print(f"  Status: {data['status_detail']}")
            else:
                print(f"  Win probability not yet available")
    else:
        print("No games found today")

    # Test injury enhancer
    print("\n\n2. Enhanced Injury Data:")
    print("-" * 40)

    injury_enhancer = ESPNInjuryEnhancer()

    test_teams = ["Los Angeles Lakers", "Boston Celtics"]
    for team in test_teams:
        injuries = injury_enhancer.get_enhanced_injuries(team)
        print(f"\n{team}:")
        print(f"  Total Out: {injuries['total_out']}")
        print(f"  Total Injured: {injuries['total_injured']}")
        for inj in injuries['injuries'][:2]:  # Show first 2
            print(f"  - {inj['status']}: {inj['injury_type']} ({inj['detail']})")
            if inj.get('return_date'):
                print(f"    Expected Return: {inj['return_date']}")

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)
