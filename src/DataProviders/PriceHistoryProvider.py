"""
Price History Provider for Polymarket CLOB

Fetches historical price data from Polymarket's CLOB API to calculate
price movements over time. Used to detect market-moving information
(injuries, lineup changes, sharp money) that our model doesn't know about.
"""

import requests
import time
from datetime import datetime, timezone


class PriceHistoryProvider:
    """Fetches price history from Polymarket CLOB API."""

    CLOB_API_URL = "https://clob.polymarket.com"

    # Cache for price history (token_id -> (timestamp, history))
    _cache = {}
    CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        pass

    def _is_cache_valid(self, cache_entry) -> bool:
        """Check if a cache entry is still valid."""
        if cache_entry is None:
            return False
        timestamp, _ = cache_entry
        return (time.time() - timestamp) < self.CACHE_TTL

    def get_price_history(self, token_id: str) -> list:
        """
        Fetch price history for a token from CLOB API.

        Args:
            token_id: The CLOB token ID

        Returns:
            List of {t: timestamp, p: price} dicts, oldest first
        """
        if not token_id:
            return []

        # Check cache
        if token_id in self._cache and self._is_cache_valid(self._cache[token_id]):
            return self._cache[token_id][1]

        try:
            response = requests.get(
                f"{self.CLOB_API_URL}/prices-history",
                params={"market": token_id, "interval": "max"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            history = data.get("history", [])

            # Cache the result
            self._cache[token_id] = (time.time(), history)
            return history

        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch price history: {e}")
            return []

    def get_price_delta(self, token_id: str, hours: int = 24) -> dict:
        """
        Calculate price change over specified time period.

        Args:
            token_id: The CLOB token ID
            hours: Number of hours to look back (default 24)

        Returns:
            dict with:
                - current_price: Current price
                - past_price: Price from `hours` ago
                - delta: Price change (current - past)
                - delta_pct: Percentage change
                - data_points: Number of data points in history
        """
        history = self.get_price_history(token_id)

        if not history or len(history) < 2:
            return {
                'current_price': None,
                'past_price': None,
                'delta': 0.0,
                'delta_pct': 0.0,
                'data_points': len(history)
            }

        current = history[-1]
        current_price = current['p']
        current_ts = current['t']

        # Find price from `hours` ago
        target_ts = current_ts - (hours * 60 * 60)

        # Find closest data point to target timestamp
        closest = min(history, key=lambda x: abs(x['t'] - target_ts))
        past_price = closest['p']

        delta = current_price - past_price
        delta_pct = delta  # Already in probability units (0-1)

        return {
            'current_price': current_price,
            'past_price': past_price,
            'delta': delta,
            'delta_pct': delta_pct,
            'data_points': len(history)
        }

    def get_market_delta(self, home_token_id: str, away_token_id: str, hours: int = 24) -> dict:
        """
        Get price delta for both sides of a market.

        Args:
            home_token_id: Token ID for home team
            away_token_id: Token ID for away team
            hours: Hours to look back

        Returns:
            dict with home and away deltas
        """
        home_delta = self.get_price_delta(home_token_id, hours)
        away_delta = self.get_price_delta(away_token_id, hours)

        return {
            'home': home_delta,
            'away': away_delta,
            'hours': hours
        }


def calculate_delta_adjustment(
    model_prob: float,
    market_prob: float,
    delta: float,
    delta_threshold: float = 0.05,
    max_blend: float = 0.5
) -> tuple:
    """
    Adjust model probability based on market price movement.

    If the market moved significantly, blend our model probability
    toward the market's current view.

    Args:
        model_prob: Our model's probability estimate
        market_prob: Current market probability
        delta: Price change over lookback period
        delta_threshold: Threshold for starting to blend (default 5%)
        max_blend: Maximum blend toward market (default 50%)

    Returns:
        (adjusted_prob, blend_factor, adjustment_reason)
    """
    abs_delta = abs(delta)

    if abs_delta < delta_threshold:
        # Small move, trust our model
        return model_prob, 0.0, "stable"

    # Calculate blend factor: scales from 0 at threshold to max_blend at 2x threshold
    # e.g., if threshold=5%, max_blend=50%:
    #   - 5% move -> 0% blend
    #   - 7.5% move -> 25% blend
    #   - 10%+ move -> 50% blend
    blend_factor = min((abs_delta - delta_threshold) / delta_threshold, 1.0) * max_blend

    # Blend model probability toward market probability
    adjusted_prob = model_prob * (1 - blend_factor) + market_prob * blend_factor

    # Determine reason
    if abs_delta >= 0.10:
        reason = "large_move"
    else:
        reason = "moderate_move"

    return adjusted_prob, blend_factor, reason


if __name__ == "__main__":
    # Test the provider
    print("Price History Provider Test")
    print("=" * 50)

    # Test with a sample token (you'd need an active market token)
    provider = PriceHistoryProvider()

    # Example calculation
    print("\nDelta Adjustment Examples:")
    print("-" * 50)

    # Example 1: Small move
    model, market, delta = 0.55, 0.53, -0.02
    adj, blend, reason = calculate_delta_adjustment(model, market, delta)
    print(f"Model: {model:.1%}, Market: {market:.1%}, Delta: {delta:+.1%}")
    print(f"  -> Adjusted: {adj:.1%}, Blend: {blend:.0%}, Reason: {reason}")

    # Example 2: Moderate move
    model, market, delta = 0.55, 0.48, -0.07
    adj, blend, reason = calculate_delta_adjustment(model, market, delta)
    print(f"\nModel: {model:.1%}, Market: {market:.1%}, Delta: {delta:+.1%}")
    print(f"  -> Adjusted: {adj:.1%}, Blend: {blend:.0%}, Reason: {reason}")

    # Example 3: Large move
    model, market, delta = 0.55, 0.42, -0.13
    adj, blend, reason = calculate_delta_adjustment(model, market, delta)
    print(f"\nModel: {model:.1%}, Market: {market:.1%}, Delta: {delta:+.1%}")
    print(f"  -> Adjusted: {adj:.1%}, Blend: {blend:.0%}, Reason: {reason}")
