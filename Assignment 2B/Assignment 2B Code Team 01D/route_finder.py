"""
COS30019 Assignment 2B: Traffic-Based Route Guidance System
Route Finder Module

Builds a road network graph from the 40 Boroondara SCATS sites and
finds the top-k shortest-time routes between an origin and destination
using Yen's k-shortest paths algorithm with A* as the underlying
single-source shortest-path solver.

Edge costs are dynamic travel times (seconds) derived from ML-predicted
traffic flow via travel_time.py.

Team ID: 01D
Team Members: Saniru, Haresh, Manula

Usage (standalone):
    python route_finder.py
"""

import math
import heapq
import copy
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from travel_time import estimate_travel_time, haversine_km

# ─────────────
# CONSTANTS
# ─────────────

BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

MAX_EDGE_KM   = 4.0    # only connect SCATS sites within this distance
K_ROUTES      = 5      # number of alternative routes to return
SEQUENCE_LEN  = 12     # must match data_processor.py


# ──────────────────────
# GRAPH CONSTRUCTION
# ──────────────────────

def _parse_primary_road(location: str) -> str:
    """
    Extract the primary road name from a SCATS location string.
    e.g. 'BURKE_RD N of HARP_RD'  →  'BURKE_RD'
         'CANTERBURY_RD E of STANHOPE_GV'  →  'CANTERBURY_RD'
    """
    # Direction keywords that separate primary road from cross road
    for sep in [" N of ", " S of ", " E of ", " W of ",
                " NE of ", " NW of ", " SE of ", " SW of ",
                " N OF ", " S OF ", " E OF ", " W OF ",
                " NE OF ", " NW OF ", " SE OF ", " SW OF "]:
        if sep in location:
            return location.split(sep)[0].strip()
    return location.strip()


def _parse_cross_road(location: str) -> str:
    """
    Extract the cross-road name from a SCATS location string.
    e.g. 'BURKE_RD N of HARP_RD'  →  'HARP_RD'
    """
    for sep in [" N of ", " S of ", " E of ", " W of ",
                " NE of ", " NW of ", " SE of ", " SW of ",
                " N OF ", " S OF ", " E OF ", " W OF ",
                " NE OF ", " NW OF ", " SE OF ", " SW OF "]:
        if sep in location:
            return location.split(sep)[1].strip()
    return ""


def build_graph(site_info: pd.DataFrame) -> dict:
    """
    Construct a directed adjacency graph for the Boroondara SCATS network.

    Edge creation rules:
      1. Sites sharing a primary road name are connected in geographic
         order along that road (adjacent pairs only).
      2. Sites where one site's primary road matches another site's
         cross-road are connected (intersection links).
      3. Any remaining isolated site is connected to its nearest
         geographic neighbour (ensures full connectivity).

    Returns:
        graph  dict  {site_id: [(neighbour_id, distance_km), ...]}
        coords dict  {site_id: (lat, lon)}
    """
    # Build lookup structures
    sites  = site_info.copy()
    coords = {
        row["scats_id"]: (row["latitude"], row["longitude"])
        for _, row in sites.iterrows()
    }
    site_ids = list(coords.keys())

    graph = defaultdict(list)

    # ── Rule 1: same primary road, adjacent pairs ──────────────────────
    sites["primary_road"] = sites["location"].apply(_parse_primary_road)
    sites["cross_road"]   = sites["location"].apply(_parse_cross_road)

    for road, group in sites.groupby("primary_road"):
        if len(group) < 2:
            continue
        # Sort by longitude first (E–W roads), then latitude (N–S roads)
        lat_spread = group["latitude"].max()  - group["latitude"].min()
        lon_spread = group["longitude"].max() - group["longitude"].min()
        sort_col   = "longitude" if lon_spread >= lat_spread else "latitude"
        group_sorted = group.sort_values(sort_col)
        ids = group_sorted["scats_id"].tolist()

        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            lat1, lon1 = coords[a]
            lat2, lon2 = coords[b]
            dist = haversine_km(lat1, lon1, lat2, lon2)
            if dist <= MAX_EDGE_KM:
                # Bidirectional
                graph[a].append((b, dist))
                graph[b].append((a, dist))

    # ── Rule 2: intersection links (primary of A == cross of B) ────────
    for _, row_a in sites.iterrows():
        for _, row_b in sites.iterrows():
            a, b = row_a["scats_id"], row_b["scats_id"]
            if a == b:
                continue
            # a's primary road is b's cross road → they share an intersection
            if (row_a["primary_road"] == row_b["cross_road"] or
                    row_b["primary_road"] == row_a["cross_road"]):
                lat1, lon1 = coords[a]
                lat2, lon2 = coords[b]
                dist = haversine_km(lat1, lon1, lat2, lon2)
                if dist <= MAX_EDGE_KM:
                    if (b, dist) not in graph[a]:
                        graph[a].append((b, dist))
                    if (a, dist) not in graph[b]:
                        graph[b].append((a, dist))

    # ── Rule 3: ensure every site has at least one edge ─────────────────
    for sid in site_ids:
        if len(graph[sid]) == 0:
            # Connect to nearest site
            best_dist, best_nbr = float("inf"), None
            lat1, lon1 = coords[sid]
            for other in site_ids:
                if other == sid:
                    continue
                lat2, lon2 = coords[other]
                d = haversine_km(lat1, lon1, lat2, lon2)
                if d < best_dist:
                    best_dist, best_nbr = d, other
            if best_nbr:
                graph[sid].append((best_nbr, best_dist))
                graph[best_nbr].append((sid, best_dist))

    # Deduplicate edges
    for sid in graph:
        seen = {}
        for (nbr, dist) in graph[sid]:
            if nbr not in seen or seen[nbr] > dist:
                seen[nbr] = dist
        graph[sid] = [(nbr, dist) for nbr, dist in seen.items()]

    # ── Rule 4: bridge disconnected components ─────────────────────────
    def _get_components(g, nodes):
        visited, components = set(), []
        for node in nodes:
            if node not in visited:
                comp, queue = set(), [node]
                while queue:
                    n = queue.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.add(n)
                    for nbr, _ in g.get(n, []):
                        queue.append(nbr)
                components.append(comp)
        return components

    for _ in range(10):
        components = _get_components(dict(graph), site_ids)
        if len(components) == 1:
            break
        main_comp = max(components, key=len)
        for comp in components:
            if comp is main_comp:
                continue
            best_d, best_a, best_b = float("inf"), None, None
            for a in main_comp:
                for b in comp:
                    lat1, lon1 = coords[a]
                    lat2, lon2 = coords[b]
                    d = haversine_km(lat1, lon1, lat2, lon2)
                    if d < best_d:
                        best_d, best_a, best_b = d, a, b
            if best_a and best_b:
                graph[best_a].append((best_b, best_d))
                graph[best_b].append((best_a, best_d))

    total_edges = sum(len(v) for v in graph.values())
    print(f"  Graph: {len(site_ids)} nodes, {total_edges} directed edges")
    return dict(graph), coords


# ────────────────────────
# DYNAMIC TRAVEL TIMES
# ────────────────────────

def apply_travel_times(graph: dict,
                       coords: dict,
                       flow_lookup: dict) -> dict:
    """
    Convert the distance-based graph into a travel-time-based graph.

    flow_lookup: {site_id: flow_15min (vehicles / 15 min)}

    Returns:
        timed_graph  {site_id: [(neighbour_id, travel_time_s), ...]}
    """
    timed_graph = {}
    for src, neighbours in graph.items():
        timed_graph[src] = []
        for (dst, dist_km) in neighbours:
            # Per spec: flow at DESTINATION site determines speed
            flow = flow_lookup.get(dst, 0.0)
            tt   = estimate_travel_time(flow, dist_km,
                                        include_intersection_delay=True)
            timed_graph[src].append((dst, tt))
    return timed_graph


# ────────────────────
# A* SHORTEST PATH
# ────────────────────

def heuristic_time(a: str, b: str, coords: dict,
                   assumed_speed_kmh: float = 60.0) -> float:
    """
    Admissible heuristic: straight-line travel time at speed limit.
    Underestimates actual travel time (no congestion, no delay).
    """
    lat1, lon1 = coords[a]
    lat2, lon2 = coords[b]
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return (dist_km / assumed_speed_kmh) * 3600.0


def astar(timed_graph: dict, coords: dict,
          origin: str, destination: str,
          forbidden_nodes: set = None,
          forbidden_edges: set = None) -> tuple:
    """
    A* search on the travel-time graph.

    forbidden_nodes / forbidden_edges: used by Yen's algorithm to
    enumerate alternative routes.

    Returns (cost, path) where path is a list of site IDs,
    or (inf, []) if no path exists.
    """
    if forbidden_nodes is None:
        forbidden_nodes = set()
    if forbidden_edges is None:
        forbidden_edges = set()

    # heap entries: (f, g, current, path)
    h0   = heuristic_time(origin, destination, coords)
    heap = [(h0, 0.0, origin, [origin])]
    best = {}   # best g-cost seen per node

    while heap:
        f, g, cur, path = heapq.heappop(heap)

        if cur in best and best[cur] <= g:
            continue
        best[cur] = g

        if cur == destination:
            return g, path

        for (nbr, cost) in timed_graph.get(cur, []):
            if nbr in forbidden_nodes:
                continue
            if (cur, nbr) in forbidden_edges:
                continue
            new_g = g + cost
            if nbr in best and best[nbr] <= new_g:
                continue
            h = heuristic_time(nbr, destination, coords)
            heapq.heappush(heap, (new_g + h, new_g, nbr,
                                  path + [nbr]))

    return float("inf"), []


# ──────────────────────────
# YEN'S K-SHORTEST PATHS
# ──────────────────────────

def yen_k_shortest(timed_graph: dict, coords: dict,
                   origin: str, destination: str,
                   k: int = K_ROUTES) -> list:
    """
    Yen's algorithm for the k-shortest loopless paths.

    Returns a list of up to k dicts:
        {"rank": int, "cost_s": float, "path": [site_ids]}

    Reference: Yen, J.Y. (1971). Finding the K shortest loopless
    paths in a network. Management Science, 17(11), 712–719.
    """
    # Find the 1st shortest path
    cost, path = astar(timed_graph, coords, origin, destination)
    if not path:
        return []

    A = [{"cost_s": cost, "path": path}]   # confirmed k-shortest paths
    B = []                                  # candidate heap

    for kk in range(1, k):
        prev = A[kk - 1]["path"]

        for i in range(len(prev) - 1):
            spur_node    = prev[i]
            root_path    = prev[: i + 1]
            root_cost    = _path_cost(root_path, timed_graph)

            # Forbidden edges: edges used by already-found paths
            # that share the same root_path
            forbidden_edges = set()
            for confirmed in A:
                cp = confirmed["path"]
                if (len(cp) > i and
                        cp[: i + 1] == root_path):
                    forbidden_edges.add((cp[i], cp[i + 1]))

            # Forbidden nodes: all nodes in root_path except spur_node
            forbidden_nodes = set(root_path[:-1])

            spur_cost, spur_path = astar(
                timed_graph, coords,
                spur_node, destination,
                forbidden_nodes=forbidden_nodes,
                forbidden_edges=forbidden_edges,
            )

            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = root_cost + spur_cost

                # Avoid duplicates
                candidate = {"cost_s": total_cost, "path": total_path}
                if not any(item[2] == total_path for item in B):
                    heapq.heappush(B, (total_cost,
                                       len(total_path),
                                       total_path))

        if not B:
            break

        _, _, best_path = heapq.heappop(B)
        best_cost = _path_cost(best_path, timed_graph)
        A.append({"cost_s": best_cost, "path": best_path})

    # Format output
    results = []
    for rank, item in enumerate(A, 1):
        results.append({
            "rank":     rank,
            "cost_s":   round(item["cost_s"], 1),
            "cost_min": round(item["cost_s"] / 60.0, 2),
            "path":     item["path"],
            "hops":     len(item["path"]) - 1,
        })
    return results


def _path_cost(path: list, timed_graph: dict) -> float:
    """Sum edge costs along a path."""
    total = 0.0
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        edge = next((c for (n, c) in timed_graph.get(src, [])
                     if n == dst), None)
        if edge is None:
            return float("inf")
        total += edge
    return total


# ─────────────────────────────────────────────
# DEFAULT FLOW LOOKUP (from saved ML predictions)
# ─────────────────────────────────────────────

def load_default_flow(site_ids: list,
                      interval_index: int = 36) -> dict:
    """
    Build a default flow_lookup from the saved per-site test data.
    interval_index=36 corresponds to 09:00 (9 AM, peak hour).

    Falls back to 0 if no saved data is found.
    """
    pkl_path = os.path.join(DATA_DIR, "per_site_data.pkl")
    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")

    flow_lookup = {sid: 0.0 for sid in site_ids}

    if not os.path.exists(pkl_path) or not os.path.exists(scaler_path):
        return flow_lookup

    with open(pkl_path, "rb") as f:
        per_site = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    for sid in site_ids:
        if sid in per_site:
            y_test = per_site[sid]["y_test"]
            idx    = min(interval_index, len(y_test) - 1)
            flow_norm = float(y_test[idx])
            flow_raw  = scaler.inverse_transform(
                [[flow_norm]])[0][0]
            flow_lookup[sid] = max(0.0, flow_raw)

    return flow_lookup


def get_flow_from_model(model, scaler, per_site_data: dict,
                        site_ids: list,
                        interval_index: int = 36) -> dict:
    """
    Run inference with a trained Keras model to get predicted flow
    for each site at a given 15-minute interval index.
    """
    flow_lookup = {}
    for sid in site_ids:
        if sid not in per_site_data:
            flow_lookup[sid] = 0.0
            continue
        X_test = per_site_data[sid]["X_test"]
        idx    = min(interval_index, len(X_test) - 1)
        x      = X_test[idx].reshape(1, -1, 1)
        pred   = model.predict(x, verbose=0)[0][0]
        flow   = float(scaler.inverse_transform([[pred]])[0][0])
        flow_lookup[sid] = max(0.0, flow)
    return flow_lookup


# ──────────────────────
# PUBLIC INTERFACE
# ──────────────────────

def find_routes(origin: str, destination: str,
                flow_lookup: dict = None,
                k: int = K_ROUTES) -> tuple:
    """
    Main entry point used by the GUI.

    Parameters
    ----------
    origin, destination : str
        SCATS site IDs (zero-padded 4-digit strings, e.g. '2000').
    flow_lookup : dict {site_id: flow_15min}, optional
        If None, uses saved test-set predictions for 09:00.
    k : int
        Number of routes to return (default 5).

    Returns
    -------
    routes : list of route dicts (see yen_k_shortest)
    site_info : pd.DataFrame  with site metadata
    """
    site_info = pd.read_csv(os.path.join(DATA_DIR, "site_info.csv"))
    site_info["scats_id"] = site_info["scats_id"].astype(str).str.zfill(4)

    graph, coords = build_graph(site_info)

    if flow_lookup is None:
        flow_lookup = load_default_flow(list(coords.keys()))

    timed_graph = apply_travel_times(graph, coords, flow_lookup)
    routes      = yen_k_shortest(timed_graph, coords,
                                  origin, destination, k=k)
    return routes, site_info


# ────────────────
# SELF-TEST
# ────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Route Finder — self test")
    print("=" * 55)

    # Use the example from the assignment spec:
    # O = 2000 (WARRIGAL_RD/TOORAK_RD)
    # D = 3002 (DENMARK_ST/BARKERS_RD)
    ORIGIN = "2000"
    DEST   = "3002"

    print(f"\n  Origin:      {ORIGIN}")
    print(f"  Destination: {DEST}")
    print(f"  Finding top-{K_ROUTES} routes …\n")

    routes, site_info = find_routes(ORIGIN, DEST)

    if not routes:
        print("  No routes found.")
    else:
        loc = dict(zip(site_info["scats_id"], site_info["location"]))
        for r in routes:
            path_str = " → ".join(r["path"])
            print(f"  Route {r['rank']}: {r['cost_min']:.1f} min "
                  f"({r['hops']} hops)   {path_str}")
