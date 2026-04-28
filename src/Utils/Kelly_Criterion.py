def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds >= 100:
        decimal_odds = (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds))
    return round(decimal_odds, 2)


def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet.

    Returns the raw Kelly percentage (e.g., 5.0 means 5% of bankroll).
    """
    decimal_odds = american_to_decimal(american_odds)
    bankroll_fraction = round((100 * (decimal_odds * model_prob - (1 - model_prob))) / decimal_odds, 2)
    return bankroll_fraction if bankroll_fraction > 0 else 0


def calculate_edge(model_prob, market_prob):
    """
    Calculates the edge as model probability minus market probability.

    Args:
        model_prob: Model's predicted probability (0-1)
        market_prob: Market's implied probability (0-1)

    Returns:
        Edge as a decimal (e.g., 0.05 = 5% edge)
    """
    return model_prob - market_prob


def calculate_tiered_kelly(
    american_odds,
    model_prob,
    market_prob,
    kelly_fraction=0.25,
    max_bet_pct=10.0
):
    """
    Calculates tiered Kelly sizing based on edge magnitude.

    Smaller edges get more conservative sizing, larger edges get more aggressive.
    This helps reduce variance while still capitalizing on high-edge opportunities.

    Args:
        american_odds: American odds for the bet
        model_prob: Model's predicted probability (0-1)
        market_prob: Market's implied probability (0-1)
        kelly_fraction: Base fractional Kelly multiplier (default 0.25 = quarter Kelly)
        max_bet_pct: Maximum bet size as percentage of bankroll (default 10%)

    Returns:
        Bet size as percentage of bankroll (0-max_bet_pct)

    Edge Tiers:
        - < 5%:  Skip (return 0)
        - 5-7%:  Conservative (25% of base Kelly)
        - 7-10%: Moderate (35% of base Kelly)
        - 10%+:  Aggressive (50% of base Kelly)
    """
    edge = calculate_edge(model_prob, market_prob)

    # Skip marginal edges
    if edge < 0.05:
        return 0

    # Calculate base Kelly
    base_kelly = calculate_kelly_criterion(american_odds, model_prob)
    if base_kelly <= 0:
        return 0

    # Apply tiered multiplier based on edge
    if edge < 0.07:
        # Conservative tier: 25% of fractional Kelly
        tier_multiplier = 0.25
    elif edge < 0.10:
        # Moderate tier: 35% of fractional Kelly
        tier_multiplier = 0.35
    else:
        # Aggressive tier: 50% of fractional Kelly
        tier_multiplier = 0.50

    # Apply fractional Kelly and tier
    bet_size = base_kelly * kelly_fraction * tier_multiplier

    # Cap at max bet percentage
    bet_size = min(bet_size, max_bet_pct)

    return round(bet_size, 2)