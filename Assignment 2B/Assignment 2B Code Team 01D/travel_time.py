"""
COS30019 Assignment 2B: Traffic-Based Route Guidance System
Travel Time Estimation Module

Converts predicted 15-minute traffic flow (vehicles/15 min) to
estimated travel time (seconds) for a road segment between two
SCATS sites, using the quadratic flow-speed model from:

    Traffic Flow to Travel Time Conversion v1.0 (VicRoads / Swinburne)

Assumptions (as per assignment specification):
  (i)   Speed limit on every link = 60 km/h
  (ii)  Flow-speed relationship: flow = -1.4648375 * speed² + 93.75 * speed
  (iii) 30-second average delay per controlled intersection

Team ID: 01D
Team Members: Saniru, Haresh, Manula
"""

import math
import numpy as np


# ─────────────────────────────────────────────
# QUADRATIC FLOW-SPEED MODEL CONSTANTS
# ─────────────────────────────────────────────
# Derived from the fundamental diagram at capacity point (1500 veh/hr, 32 km/h):
#   A = -v_c / q_c^2  = -32 / 1500^2  → adjusted to -1.4648375 (as per doc)
#   B = -2 * v_c / q_c * A            → adjusted to 93.75      (as per doc)
#
# Forward equation:  flow = A * speed² + B * speed
# Inverse (speed from flow via quadratic formula):
#   speed = (-B ± sqrt(B² + 4*A*flow)) / (2*A)

A = -1.4648375   # coefficient of speed²
B = 93.75        # coefficient of speed
SPEED_LIMIT_KMH   = 60.0    # km/h
INTERSECTION_DELAY_S = 30.0  # seconds per intersection
# Flow threshold below which speed equals the speed limit (351 veh/hr)
FREE_FLOW_THRESHOLD_VPH = 351.0


# ─────────────────────────────────────────────
# CORE CONVERSION FUNCTIONS
# ─────────────────────────────────────────────

def flow_per15min_to_vph(flow_15: float) -> float:
    """
    Convert a 15-minute vehicle count to vehicles-per-hour.
    (Multiply by 4.)
    """
    return float(flow_15) * 4.0


def vph_to_speed_kmh(flow_vph: float) -> float:
    """
    Convert hourly flow (vehicles/hour) to expected speed (km/h)
    using the quadratic flow-speed model.

    Two branches:
      Green (free-flow):  flow ≤ 351 veh/hr  → speed = 60 km/h (capped at limit)
      Red   (congested):  flow  > 351 veh/hr  → solve quadratic, take lower root

    Returns speed in km/h (always positive, floored at 1 km/h to avoid div/0).
    """
    if flow_vph <= 0:
        return SPEED_LIMIT_KMH

    if flow_vph <= FREE_FLOW_THRESHOLD_VPH:
        # Road is under capacity — speed equals or exceeds the limit
        return SPEED_LIMIT_KMH

    # Congested branch: solve A*speed² + B*speed - flow = 0
    # ⟹  speed = (-B - sqrt(B² + 4*A*flow)) / (2*A)
    # Note: because A < 0 the "−" root gives the lower (congested) speed
    discriminant = B ** 2 + 4 * A * flow_vph
    if discriminant < 0:
        # Flow beyond the parabola's domain — road is completely jammed
        return 1.0   # 1 km/h floor

    speed = (-B - math.sqrt(discriminant)) / (2 * A)
    # Clamp to a sensible range [1, SPEED_LIMIT]
    speed = max(1.0, min(speed, SPEED_LIMIT_KMH))
    return speed


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two coordinates (in kilometres).
    Used to estimate road segment length between adjacent SCATS sites.
    """
    R = 6371.0   # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_travel_time(
        flow_15min: float,
        distance_km: float,
        include_intersection_delay: bool = True) -> float:
    """
    Estimate travel time in seconds for one road segment.

    Parameters
    ----------
    flow_15min : float
        Predicted traffic volume at the destination SCATS site
        for the relevant 15-minute interval (vehicles / 15 min).
    distance_km : float
        Road segment length in kilometres (Haversine approximation).
    include_intersection_delay : bool
        Whether to add the 30-second intersection delay
        (True for all intermediate nodes; can be False for the
        destination to avoid double-counting).

    Returns
    -------
    float  Travel time in seconds (minimum 1 s).
    """
    flow_vph = flow_per15min_to_vph(flow_15min)
    speed_kmh = vph_to_speed_kmh(flow_vph)

    # time = distance / speed  (both in km and km/h → result in hours → ×3600 for seconds)
    travel_s = (distance_km / speed_kmh) * 3600.0

    if include_intersection_delay:
        travel_s += INTERSECTION_DELAY_S

    return max(1.0, travel_s)


# ──────────────────────
# GRAPH-LEVEL HELPERS
# ──────────────────────

def build_edge_travel_times(site_info,
                             edge_list: list,
                             flow_lookup: dict) -> dict:
    """
    Compute travel time for every edge in the SCATS road graph.

    Parameters
    ----------
    site_info : pd.DataFrame
        Columns: scats_id, latitude, longitude
    edge_list : list of (from_id, to_id)
        All directed edges in the network.
    flow_lookup : dict  {scats_id: float}
        Latest predicted 15-min flow per site.

    Returns
    -------
    dict  {(from_id, to_id): travel_time_seconds}
    """
    coords = {
        row["scats_id"]: (row["latitude"], row["longitude"])
        for _, row in site_info.iterrows()
    }

    edge_times = {}
    for (src, dst) in edge_list:
        if src not in coords or dst not in coords:
            continue
        lat1, lon1 = coords[src]
        lat2, lon2 = coords[dst]
        dist_km = haversine_km(lat1, lon1, lat2, lon2)

        # Per spec: flow is taken from the DESTINATION site
        flow = flow_lookup.get(dst, 0.0)
        tt   = estimate_travel_time(flow, dist_km)
        edge_times[(src, dst)] = tt

    return edge_times


def get_speed_category(flow_vph: float) -> str:
    """Return a human-readable traffic condition label."""
    if flow_vph <= FREE_FLOW_THRESHOLD_VPH:
        return "Free flow"
    elif flow_vph <= 900:
        return "Moderate"
    elif flow_vph <= 1200:
        return "Heavy"
    else:
        return "Congested"


# ───────────────────
# QUICK SELF-TEST
# ───────────────────

if __name__ == "__main__":
    print("Travel Time Conversion – self test")
    print("=" * 50)

    test_cases = [
        ("Zero flow",        0,    2.0),
        ("Low flow (100)",  100,   2.0),
        ("Threshold (351)", 351,   2.0),
        ("Moderate (600)",  600,   2.0),
        ("Heavy (1000)",   1000,   2.0),
        ("Capacity (1500)", 1500,  2.0),
    ]

    print(f"  {'Scenario':<22} {'Flow(15min)':>12} {'Vph':>6} "
          f"{'Speed(km/h)':>12} {'TT(s)':>8} {'Category'}")
    print("  " + "-" * 75)
    for label, flow15, dist in test_cases:
        vph   = flow_per15min_to_vph(flow15)
        speed = vph_to_speed_kmh(vph)
        tt    = estimate_travel_time(flow15, dist)
        cat   = get_speed_category(vph)
        print(f"  {label:<22} {flow15:>12.0f} {vph:>6.0f} "
              f"{speed:>12.1f} {tt:>8.1f}  {cat}")
