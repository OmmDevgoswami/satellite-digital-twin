"""
severity.py
-----------
Severity scoring and environmental impact estimation for detected dump sites.

WHY severity scores?
  A binary dump/no-dump decision is not enough for actionable policy.
  We combine classifier confidence, spatial coverage, and estimated area
  into a 0-100 severity index that prioritises cleanup intervention.

Severity Formula:
  S = 0.4 * prob + 0.4 * (coverage/100) + 0.2 * clamp(area/10, 0, 1)
  Scaled 0→100 and bucketed: LOW / MEDIUM / HIGH / CRITICAL
"""

from __future__ import annotations


# ── Severity Score ────────────────────────────────────────────────────────────
def compute_severity_score(
    prob: float,
    coverage_pct: float,
    area_ha: float = 2.0,
) -> dict:
    """
    Compute a 0–100 severity score for a detected dump site.

    Args:
        prob         : classifier probability [0, 1]
        coverage_pct : % of image covered by dump mask [0, 100]
        area_ha      : estimated physical area in hectares (default 2 ha)

    Returns:
        dict with 'score' (float), 'level' (str), 'color' (str), 'emoji' (str)
    """
    prob        = float(max(0.0, min(1.0, prob)))
    coverage    = float(max(0.0, min(100.0, coverage_pct))) / 100.0
    area_norm   = float(min(area_ha / 10.0, 1.0))          # normalise to [0,1]

    raw_score   = 0.40 * prob + 0.40 * coverage + 0.20 * area_norm
    score       = round(raw_score * 100, 1)

    if score >= 75:
        level, color, emoji = "CRITICAL", "#c0392b", "🔴"
    elif score >= 50:
        level, color, emoji = "HIGH",     "#e67e22", "🟠"
    elif score >= 25:
        level, color, emoji = "MEDIUM",   "#f1c40f", "🟡"
    else:
        level, color, emoji = "LOW",      "#27ae60", "🟢"

    return {
        "score": score,
        "level": level,
        "color": color,
        "emoji": emoji,
    }


# ── Environmental Impact ──────────────────────────────────────────────────────
_WASTE_DENSITY_TONNES_PER_HA = 150.0   # avg. density for mixed waste dumps
_CO2_PER_TONNE_KG            = 1.2     # CH4 + CO2 estimate for open dumps
_CLEANUP_COST_INR_PER_TONNE  = 2200.0  # avg municipal cleanup ₹/tonne (India)


def estimate_environmental_impact(
    coverage_pct: float,
    area_ha: float = 2.0,
) -> dict:
    """
    Estimate environmental impact metrics for a detected dump site.

    Args:
        coverage_pct : % of image area covered by dump mask
        area_ha      : estimated physical area in hectares

    Returns:
        dict with estimated_area_ha, tonnes_waste, CO2_tonnes, cleanup_cost_inr
    """
    # Scale area by coverage fraction for a conservative estimate
    eff_area_ha  = area_ha * (coverage_pct / 100.0)

    tonnes_waste  = round(eff_area_ha * _WASTE_DENSITY_TONNES_PER_HA, 1)
    co2_tonnes    = round(tonnes_waste * _CO2_PER_TONNE_KG / 1000.0, 2)
    cleanup_cost  = round(tonnes_waste * _CLEANUP_COST_INR_PER_TONNE, 0)

    return {
        "estimated_area_ha":  round(eff_area_ha, 2),
        "tonnes_waste":       tonnes_waste,
        "CO2_tonnes":         co2_tonnes,
        "cleanup_cost_inr":   int(cleanup_cost),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sv = compute_severity_score(0.88, 42.0, 3.5)
    imp = estimate_environmental_impact(42.0, 3.5)
    print("Severity:", sv)
    print("Impact  :", imp)
