"""
Injury-Based Probability Adjustment

Adjusts model probabilities based on injury impact differential between teams.
"""


def calculate_injury_adjustment(
    model_prob: float,
    team_impact: float,
    opponent_impact: float,
    max_adjustment: float = 0.10,
    adjustment_factor: float = 0.025
) -> float:
    """
    Calculate adjusted probability based on injury impacts.

    The adjustment is based on the net injury impact between teams:
    - If opponent has more injuries (higher impact), probability increases
    - If team has more injuries (higher impact), probability decreases

    Args:
        model_prob: Original model probability (0-1)
        team_impact: Injury impact score for the team (sum of importance for OUT players)
        opponent_impact: Injury impact score for the opponent
        max_adjustment: Maximum allowed adjustment (default 10%)
        adjustment_factor: Adjustment per unit of impact (default 2.5%)

    Returns:
        Adjusted probability, capped between 0.01 and 0.99

    Example:
        - Lakers missing LeBron (35 min) + AD (34 min) = 1.44 impact
        - Celtics healthy = 0 impact
        - net_impact = 0 - 1.44 = -1.44 (bad for Lakers)
        - adjustment = -1.44 * 0.025 = -0.036 (-3.6%)
        - Model 55% -> Adjusted 51.4%
    """
    # Calculate net impact (positive = good for team, negative = bad)
    net_impact = opponent_impact - team_impact

    # Calculate adjustment
    adjustment = net_impact * adjustment_factor

    # Cap the adjustment
    adjustment = max(-max_adjustment, min(max_adjustment, adjustment))

    # Apply adjustment
    adjusted_prob = model_prob + adjustment

    # Ensure probability stays in valid range
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))

    return adjusted_prob


def format_injury_adjustment(
    team_impact: float,
    opponent_impact: float,
    adjustment_factor: float = 0.025,
    max_adjustment: float = 0.10
) -> str:
    """
    Format injury adjustment as a display string.

    Args:
        team_impact: Injury impact for this team
        opponent_impact: Injury impact for opponent
        adjustment_factor: Adjustment per unit of impact
        max_adjustment: Maximum allowed adjustment

    Returns:
        Formatted string like "+3.6%" or "-2.1%"
    """
    net_impact = opponent_impact - team_impact
    adjustment = net_impact * adjustment_factor
    adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
    return f"{adjustment:+.1%}"


if __name__ == "__main__":
    # Test examples
    print("Injury Adjustment Examples:")
    print("-" * 50)

    # Example 1: Lakers missing key players vs healthy Celtics
    print("\nExample 1: Lakers vs Celtics")
    print("  Lakers: LeBron (35 min) + AD (34 min) OUT")
    print("  Celtics: Healthy")

    lakers_impact = (35 + 34) / 48  # 1.44
    celtics_impact = 0

    print(f"  Lakers impact: {lakers_impact:.3f}")
    print(f"  Celtics impact: {celtics_impact:.3f}")

    # Lakers home win probability
    model_prob = 0.55
    adjusted = calculate_injury_adjustment(model_prob, lakers_impact, celtics_impact)
    print(f"  Model Lakers win: {model_prob:.1%}")
    print(f"  Adjusted Lakers win: {adjusted:.1%}")
    print(f"  Adjustment: {format_injury_adjustment(lakers_impact, celtics_impact)}")

    # Example 2: Both teams have injuries
    print("\nExample 2: Warriors vs Suns")
    print("  Warriors: Curry (32 min) OUT")
    print("  Suns: Booker (36 min) OUT")

    warriors_impact = 32 / 48  # 0.67
    suns_impact = 36 / 48  # 0.75

    print(f"  Warriors impact: {warriors_impact:.3f}")
    print(f"  Suns impact: {suns_impact:.3f}")

    model_prob = 0.48
    adjusted = calculate_injury_adjustment(model_prob, warriors_impact, suns_impact)
    print(f"  Model Warriors win: {model_prob:.1%}")
    print(f"  Adjusted Warriors win: {adjusted:.1%}")
    print(f"  Adjustment: {format_injury_adjustment(warriors_impact, suns_impact)}")
