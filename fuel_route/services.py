"""
fuel_route/services.py

Core routing and fuel-optimisation logic for the Spotter Fleet platform.

External dependency (one network call per unique route pair):
  OSRM-compatible driving API — primary endpoint is the OSM-hosted instance;
  the project-osrm.org public demo server is used as a fallback.

Fuel accounting model — Full-Tank Refuel
-----------------------------------------
The vehicle departs the origin with a **full tank** (50 gallons / 500 miles).
At each selected intermediate stop the driver **fills the tank to capacity**:

    gallons_pumped  = TANK_CAPACITY_GALLONS − fuel_before_arrival
    cost            = gallons_pumped × retail_price

This is the only internally consistent model when stations are spaced
unevenly: an exact-purchase model (buy only next_leg/MPG gallons) can
overflow the tank when incoming_leg < next_leg.

Fuel balance identity (verified in the test suite):
    initial_fuel + Σ gallons_pumped − total_distance / MPG = fuel_remaining

Fields returned by ``optimise_refuel_stops``:
    station_refuel_cost_usd          — sum of pump purchases at all stops
                                       ($0 for short trips within one tank)
    trip_consumption_cost_usd        — total fuel cost = station_refuel_cost_usd
                                       (last stop fills to full; terminal leg
                                       is already covered; no double-count)
    total_fuel_cost                  — backward-compatible alias
    fuel_remaining_at_destination_gal — TANK − final_leg / MPG (≥ 0)
"""

import json
import math
import logging
from pathlib import Path

import requests
import polyline as polyline_lib
import numpy as np
from scipy.spatial import cKDTree
from django.conf import settings
from geopy.geocoders import Nominatim

from fuel_route.models import TruckStop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application-level exceptions
# ---------------------------------------------------------------------------

class RoutingServiceError(Exception):
    """Raised when an upstream routing service call fails or returns no routes."""


class LocationNotFoundError(Exception):
    """Raised when a location string cannot be resolved to coordinates."""


# ---------------------------------------------------------------------------
# Vehicle & routing constants (all overridable via Django settings)
# ---------------------------------------------------------------------------

TANK_CAPACITY_MILES: float = getattr(settings, "VEHICLE_RANGE_MILES", 500.0)
MPG: float = getattr(settings, "VEHICLE_MPG", 10.0)
CORRIDOR_RADIUS_MILES: float = getattr(settings, "ROUTE_CORRIDOR_RADIUS_MILES", 10.0)

# Cluster stops within this window before running DP to avoid micro-stops
# that arise from many city-centroid stations at nearly the same route mile.
CLUSTER_WINDOW_MILES: float = getattr(settings, "ROUTE_CLUSTER_WINDOW_MILES", 2.0)

TANK_CAPACITY_GALLONS: float = TANK_CAPACITY_MILES / MPG   # 50.0

METERS_TO_MILES: float = 1.0 / 1609.344
MILES_PER_DEGREE_LAT: float = 69.172

OSRM_ENDPOINTS: list[str] = [
    "https://routing.openstreetmap.de/routed-car/route/v1/driving",
    "https://router.project-osrm.org/route/v1/driving",
]

_OSRM_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": "SpotterFuelOptimizer/1.0 (assessment@spotter.local)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_GEO_DATA: dict | None = None
_GEOCODE_CACHE: dict[str, tuple] = {}
_OSRM_CACHE: dict[tuple, tuple] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_geo_data() -> dict:
    """Lazy-loads the bundled city/state coordinate dictionary once."""
    global _GEO_DATA
    if _GEO_DATA is None:
        path = Path(__file__).resolve().parent / "data" / "city_coords.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                _GEO_DATA = json.load(fh)
            logger.debug("Loaded geo data: %d city entries", len(_GEO_DATA.get("cities", {})))
        else:
            logger.warning("city_coords.json not found at %s; offline lookup disabled", path)
            _GEO_DATA = {"cities": {}, "states": {}}
    return _GEO_DATA


def _estimate_corridor_price(candidate_stops: list[dict]) -> float:
    """Returns the median retail price from corridor candidates, or a US average fallback."""
    if not candidate_stops:
        return 3.50
    prices = sorted(s["retail_price"] for s in candidate_stops)
    return prices[len(prices) // 2]


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_location(query: str) -> tuple[float, float, str]:
    """
    Resolves a location string to ``(latitude, longitude, display_name)``.

    Resolution order:
      1. In-process cache
      2. Raw ``"lat, lon"`` passthrough
      3. Offline city/state dictionary
      4. OpenStreetMap Nominatim (network, last resort)

    Raises:
        LocationNotFoundError: when the location cannot be resolved by any method.
    """
    query = query.strip()
    if not query:
        raise LocationNotFoundError("Location query must not be blank.")

    cache_key = query.lower()
    if cache_key in _GEOCODE_CACHE:
        logger.debug("Geocode cache hit: %r", query)
        return _GEOCODE_CACHE[cache_key]

    # Raw coordinate passthrough
    if "," in query:
        parts = [p.strip() for p in query.split(",", 1)]
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                result = (lat, lon, query)
                _GEOCODE_CACHE[cache_key] = result
                return result
        except ValueError:
            pass

    geo = _load_geo_data()
    cities_db: dict = geo.get("cities", {})
    states_db: dict = geo.get("states", {})

    # City, State lookup
    if "," in query:
        city_raw, state_raw = query.split(",", 1)
        key = f"{city_raw.strip().lower()}|{state_raw.strip().upper()}"
        if key in cities_db:
            lat, lon = cities_db[key]
            result = (lat, lon, f"{city_raw.strip().title()}, {state_raw.strip().upper()}")
            _GEOCODE_CACHE[cache_key] = result
            logger.debug("Offline city lookup: %r -> (%.4f, %.4f)", query, lat, lon)
            return result

    # State centroid
    if query.upper() in states_db:
        lat, lon = states_db[query.upper()]
        result = (lat, lon, query.upper())
        _GEOCODE_CACHE[cache_key] = result
        return result

    # Nominatim fallback
    logger.info("Falling back to Nominatim for %r", query)
    try:
        geolocator = Nominatim(user_agent="spotter_fleet_router", timeout=5)
        loc = geolocator.geocode(query, country_codes="us")
        if loc:
            result = (loc.latitude, loc.longitude, loc.address)
            _GEOCODE_CACHE[cache_key] = result
            return result
    except Exception as exc:
        logger.warning("Nominatim failed for %r: %s", query, exc)

    raise LocationNotFoundError(f"Could not resolve location: {query!r}")


# ---------------------------------------------------------------------------
# OSRM routing
# ---------------------------------------------------------------------------

def fetch_driving_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> tuple[float, list[tuple[float, float]], str]:
    """
    Fetches a driving route from an OSRM-compatible API.

    Makes **exactly one** network call per unique (origin, destination) pair;
    subsequent calls with the same coordinates are served from an in-process cache.

    Args:
        origin: ``(latitude, longitude)`` of the start point.
        destination: ``(latitude, longitude)`` of the end point.

    Returns:
        A 3-tuple of ``(distance_miles, route_waypoints, encoded_polyline)``.

    Raises:
        RoutingServiceError: when all configured endpoints fail or return no routes.
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))

    if cache_key in _OSRM_CACHE:
        logger.debug("OSRM cache hit for route (%s, %s) -> (%s, %s)", lat1, lon1, lat2, lon2)
        return _OSRM_CACHE[cache_key]

    path = f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=polyline"
    last_error: Exception | None = None

    for base_url in OSRM_ENDPOINTS:
        url = f"{base_url}/{path}"
        try:
            logger.info("Calling OSRM: %s", url)
            resp = requests.get(url, headers=_OSRM_REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            if payload.get("code") != "Ok" or not payload.get("routes"):
                raise RoutingServiceError(
                    f"OSRM response code={payload.get('code')!r}: "
                    f"{payload.get('message', 'no routes returned')}"
                )

            route = payload["routes"][0]
            distance_miles = route["distance"] * METERS_TO_MILES
            encoded = route["geometry"]
            waypoints = polyline_lib.decode(encoded)

            result = (distance_miles, waypoints, encoded)
            _OSRM_CACHE[cache_key] = result
            logger.info("Route fetched: %.1f miles, %d waypoints", distance_miles, len(waypoints))
            return result

        except RoutingServiceError:
            raise
        except Exception as exc:
            logger.warning("OSRM endpoint %s failed: %s", base_url, exc)
            last_error = exc

    raise RoutingServiceError(
        f"All OSRM endpoints unavailable. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Spatial corridor filtering
# ---------------------------------------------------------------------------

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _densify_waypoints(
    waypoints: list[tuple], max_step: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates waypoints so no consecutive pair is more than *max_step* miles apart."""
    if not waypoints:
        return np.empty((0, 2)), np.empty(0)

    pts = [waypoints[0]]
    miles = [0.0]
    total = 0.0

    for i in range(1, len(waypoints)):
        prev, curr = waypoints[i - 1], waypoints[i]
        seg = _haversine_miles(prev[0], prev[1], curr[0], curr[1])
        if seg <= max_step:
            total += seg
            pts.append(curr)
            miles.append(total)
        else:
            steps = math.ceil(seg / max_step)
            for s in range(1, steps + 1):
                f = s / steps
                pts.append((prev[0] + f * (curr[0] - prev[0]),
                             prev[1] + f * (curr[1] - prev[1])))
                miles.append(total + f * seg)
            total += seg

    return np.array(pts), np.array(miles)


def _to_cartesian(coords: np.ndarray, lat_ref: float) -> np.ndarray:
    lon_scale = MILES_PER_DEGREE_LAT * math.cos(math.radians(lat_ref))
    return np.column_stack((coords[:, 1] * lon_scale, coords[:, 0] * MILES_PER_DEGREE_LAT))


def find_corridor_stops(
    waypoints: list[tuple],
    total_miles: float,
) -> list[dict]:
    """
    Returns all TruckStop records within ``CORRIDOR_RADIUS_MILES`` of the route,
    each annotated with ``mile_along_route`` (cumulative distance from origin).

    Uses a ``cKDTree`` spatial index for O(n log n) nearest-neighbour queries.
    """
    if not waypoints:
        return []

    dense_pts, dense_miles = _densify_waypoints(waypoints)
    lat_ref = float(np.mean(dense_pts[:, 0]))

    pad_lat = CORRIDOR_RADIUS_MILES / MILES_PER_DEGREE_LAT + 0.05
    pad_lon = CORRIDOR_RADIUS_MILES / (MILES_PER_DEGREE_LAT * math.cos(math.radians(lat_ref))) + 0.05

    bbox_candidates = list(
        TruckStop.objects.filter(
            latitude__gte=float(dense_pts[:, 0].min()) - pad_lat,
            latitude__lte=float(dense_pts[:, 0].max()) + pad_lat,
            longitude__gte=float(dense_pts[:, 1].min()) - pad_lon,
            longitude__lte=float(dense_pts[:, 1].max()) + pad_lon,
        ).values(
            "id", "opis_id", "name", "address",
            "city", "state", "retail_price", "latitude", "longitude",
        )
    )

    if not bbox_candidates:
        logger.info("No candidate stops within bounding box.")
        return []

    route_xy = _to_cartesian(dense_pts, lat_ref)
    tree = cKDTree(route_xy)

    stop_coords = np.array([[s["latitude"], s["longitude"]] for s in bbox_candidates])
    stop_xy = _to_cartesian(stop_coords, lat_ref)

    dists, idxs = tree.query(stop_xy, distance_upper_bound=CORRIDOR_RADIUS_MILES)

    corridor: list[dict] = []
    for i, dist in enumerate(dists):
        if dist <= CORRIDOR_RADIUS_MILES:
            mile = float(dense_miles[idxs[i]])
            if 0.0 <= mile <= total_miles:
                stop = dict(bbox_candidates[i])
                stop["mile_along_route"] = round(mile, 2)
                stop["corridor_distance_miles"] = round(float(dist), 2)
                corridor.append(stop)

    corridor.sort(key=lambda s: (s["mile_along_route"], s["retail_price"]))
    logger.info("Corridor: %d stops within %.0f mi of route", len(corridor), CORRIDOR_RADIUS_MILES)
    return corridor


# ---------------------------------------------------------------------------
# Stop deduplication / clustering
# ---------------------------------------------------------------------------

def _cluster_stops(stops_sorted: list[dict], window_miles: float) -> list[dict]:
    """
    Reduces a sorted list of corridor stops to at most one representative per
    ``window_miles`` sliding window, retaining the cheapest station in each
    cluster.

    This prevents the DP from selecting a series of micro-stops (e.g. ten
    stations 2 miles apart) that arise when many truck stops share the same
    city-centroid coordinate and are projected to nearly identical route miles.

    Args:
        stops_sorted: stops sorted ascending by ``mile_along_route``.
        window_miles: minimum separation between chosen representatives.

    Returns:
        A pruned list of stops, one per window, cheapest price wins.
    """
    if not stops_sorted:
        return []

    clusters: list[dict] = []
    window_start = -window_miles  # ensures first stop always starts a new window

    for stop in stops_sorted:
        m = stop["mile_along_route"]
        if m - window_start >= window_miles:
            # Start a new cluster with this stop
            clusters.append(stop)
            window_start = m
        else:
            # Same cluster — keep the cheaper one
            if stop["retail_price"] < clusters[-1]["retail_price"]:
                clusters[-1] = stop

    return clusters


# ---------------------------------------------------------------------------
# DAG Dynamic Programming fuel optimisation
# ---------------------------------------------------------------------------

def optimise_refuel_stops(
    candidate_stops: list[dict],
    total_distance: float,
) -> dict:
    """
    Selects the globally minimum-cost refuelling sequence using a Directed
    Acyclic Graph (DAG) Dynamic Programming algorithm under the
    **Full-Tank Refuel Model**:

    - The vehicle departs the origin with a full tank (50 gal / 500 mi).
    - At each selected stop the driver **fills the tank to capacity**.
      ``gallons_pumped = TANK_CAPACITY_GALLONS − fuel_before_arrival``
    - ``station_refuel_cost_usd`` is the sum of those pump purchases.
    - ``trip_consumption_cost_usd`` equals ``station_refuel_cost_usd``
      (the last stop fills to full, so the terminal leg is already covered).
    - ``fuel_remaining_at_destination_gal = TANK − final_leg / MPG``.

    Fuel balance identity (verified by tests):
        initial_fuel + Σ gallons_pumped − total_gallons_burned = fuel_remaining

    DAG graph construction
    ----------------------
    Nodes  : Origin (mile 0)
             + corridor stops clustered into 2-mile windows (cheapest per window)
             + Destination (total_distance).
    Edges  : u -> v exists iff  0 < mile(v) - mile(u) <= TANK_CAPACITY_MILES.
    Cost   : cost(Origin -> v)  = 0.0   (departs with pre-filled tank)
             cost(stop_u -> v)  = (leg_miles / MPG) * stop_u.retail_price

    Recurrence
    ----------
    dp[v] = min over all valid predecessors u of  dp[u] + cost(u, v)

    Predecessor pointers enable exact path reconstruction via backtracking.
    Time complexity: O(n^2) worst-case; bounded in practice by the 500-mile
    window (typically < 50 reachable predecessors per node on US highways).

    Fuel desert detection
    ---------------------
    If no path reaches the destination (gap > TANK_CAPACITY_MILES between
    consecutive viable stops), the function logs a warning and returns
    ``reachable: False``.

    Returns
    -------
    dict matching the ``RouteResponseSerializer`` schema.
    """
    total_gallons = round(total_distance / MPG, 2)
    benchmark_price = _estimate_corridor_price(candidate_stops)

    # ── Short trip (fits within one full tank) ───────────────────────────────
    if total_distance <= TANK_CAPACITY_MILES:
        trip_cost = round(total_gallons * benchmark_price, 2)
        fuel_remaining = round(TANK_CAPACITY_GALLONS - total_gallons, 2)
        return {
            "fuel_stops": [],
            "total_distance_miles": round(total_distance, 2),
            "total_fuel_gallons": total_gallons,
            "station_refuel_cost_usd": 0.0,
            "trip_consumption_cost_usd": trip_cost,
            "total_fuel_cost": trip_cost,           # backward-compat alias
            "fuel_remaining_at_destination_gal": fuel_remaining,
            "number_of_stops": 0,
            "reachable": True,
        }

    # ── Cluster stops to prevent micro-stop selection ────────────────────────
    stops_pre = sorted(candidate_stops, key=lambda s: (s["mile_along_route"], s["retail_price"]))
    stops_sorted = _cluster_stops(stops_pre, window_miles=CLUSTER_WINDOW_MILES)
    logger.info("After clustering: %d stops (was %d)", len(stops_sorted), len(candidate_stops))

    # ── Build DAG ────────────────────────────────────────────────────────────
    # Node 0 = origin (virtual), nodes 1..n = stops, node n+1 = destination
    n = len(stops_sorted)
    ORIGIN = 0
    DEST = n + 1

    def mile(idx: int) -> float:
        if idx == ORIGIN:
            return 0.0
        if idx == DEST:
            return total_distance
        return stops_sorted[idx - 1]["mile_along_route"]

    def stop_price(idx: int) -> float:
        return stops_sorted[idx - 1]["retail_price"]

    INF = float("inf")
    dp:   list[float] = [INF] * (DEST + 1)
    pred: list[int]   = [-1]  * (DEST + 1)
    dp[ORIGIN] = 0.0

    for v in range(1, DEST + 1):
        v_mile = mile(v)
        for u in range(v - 1, ORIGIN - 1, -1):
            u_mile = mile(u)
            leg = v_mile - u_mile
            if leg > TANK_CAPACITY_MILES:
                break       # all further u's have even larger legs
            if leg <= 0.0:
                continue
            if dp[u] == INF:
                continue

            # Origin edge: vehicle departs with a pre-filled tank — no cost
            edge_cost = 0.0 if u == ORIGIN else (leg / MPG) * stop_price(u)

            candidate_cost = dp[u] + edge_cost
            if candidate_cost < dp[v]:
                dp[v] = candidate_cost
                pred[v] = u

    # ── Fuel desert check ────────────────────────────────────────────────────
    if dp[DEST] == INF:
        logger.warning(
            "Route unreachable: no chain of <=%.0f-mile legs covers %.1f mi. "
            "A fuel desert gap exceeds vehicle range.",
            TANK_CAPACITY_MILES, total_distance,
        )
        return {
            "fuel_stops": [],
            "total_distance_miles": round(total_distance, 2),
            "total_fuel_gallons": total_gallons,
            "station_refuel_cost_usd": 0.0,
            "trip_consumption_cost_usd": 0.0,
            "total_fuel_cost": 0.0,
            "fuel_remaining_at_destination_gal": 0.0,
            "number_of_stops": 0,
            "reachable": False,
        }

    # ── Reconstruct optimal path ─────────────────────────────────────────────
    path: list[int] = []
    node = DEST
    while node != ORIGIN:
        path.append(node)
        node = pred[node]
    path.reverse()   # [stop_1, stop_2, ..., DEST]

    # ── Build per-stop records — Full-Tank Refuel Model ─────────────────────
    #
    # At each selected stop the driver **fills the tank to capacity** (50 gal).
    # This is internally consistent with the existing ``fuel_remaining``
    # formula (TANK − final_leg/MPG) and avoids overflow that arises when
    # ``next_leg > incoming_leg`` under an exact-purchase model.
    #
    # gallons_pumped  = TANK_CAPACITY_GALLONS − fuel_before_arrival
    # fuel_remaining  = TANK_CAPACITY_GALLONS − final_leg / MPG      (≥ 0)
    #
    # Fuel balance identity (verified in tests):
    #   initial_fuel + Σ gallons_pumped − total_gallons = fuel_remaining
    #   50 + Σ pumped − (distance / MPG) = 50 − last_leg / MPG
    #
    # DP edge cost formula (leg_miles / MPG) × price is unaffected — it
    # measures the economic cost of fuelling each leg at the departure price,
    # regardless of whether the tank is topped up further.
    stops_chosen: list[dict] = []
    station_cost = 0.0
    prev_node = ORIGIN
    fuel_level = TANK_CAPACITY_GALLONS   # start with a full tank

    for path_idx, node in enumerate(path):
        if node == DEST:
            break

        stop_data = stops_sorted[node - 1]
        leg_incoming = mile(node) - mile(prev_node)

        # Consume fuel for the incoming leg; verify reachability
        fuel_level -= leg_incoming / MPG
        fuel_level = round(fuel_level, 6)
        if fuel_level < -1e-6:
            # Should never happen given DP edge constraint, but guard anyway
            logger.error(
                "Negative fuel (%.4f gal) before stop '%s' at mile %.1f — "
                "DP path violated TANK_CAPACITY_MILES constraint.",
                fuel_level, stop_data["name"], stop_data["mile_along_route"],
            )

        # Fill to full
        fuel_before_arrival = max(0.0, fuel_level)
        gallons = round(TANK_CAPACITY_GALLONS - fuel_before_arrival, 4)
        gallons = max(0.0, gallons)
        fuel_level = TANK_CAPACITY_GALLONS

        cost = round(gallons * stop_data["retail_price"], 2)

        stops_chosen.append({
            "opis_id":            stop_data["opis_id"],
            "name":               stop_data["name"],
            "address":            stop_data["address"],
            "city":               stop_data["city"],
            "state":              stop_data["state"],
            "retail_price":       round(stop_data["retail_price"], 3),
            "latitude":           stop_data["latitude"],
            "longitude":          stop_data["longitude"],
            "mile_along_route":   stop_data["mile_along_route"],
            "leg_distance_miles": round(leg_incoming, 2),
            "gallons_pumped":     round(gallons, 4),
            "cost":               cost,
        })
        station_cost += cost
        prev_node = node

    station_cost = round(station_cost, 2)

    # ── Terminal leg and fuel balance ────────────────────────────────────────
    last_leg = total_distance - (
        stops_chosen[-1]["mile_along_route"] if stops_chosen else 0.0
    )
    # After the last stop the tank is full (50 gal); drive last_leg miles.
    fuel_remaining = round(TANK_CAPACITY_GALLONS - last_leg / MPG, 4)
    fuel_remaining = max(0.0, fuel_remaining)

    logger.info(
        "DP optimisation: %d stops, station_cost=$%.2f, fuel_remaining=%.4f gal",
        len(stops_chosen), station_cost, fuel_remaining,
    )

    # trip_consumption_cost_usd: total fuel purchased at stations.
    # Under Full-Tank model the last stop fills to 50 gal, so
    # station_cost already includes fuel for the terminal leg.
    # No separate final-leg addition is needed.
    # total_fuel_gallons = total_distance / MPG (all fuel burned; some was
    # from the initial pre-filled tank, the rest was purchased at stops).
    return {
        "fuel_stops":                        stops_chosen,
        "total_distance_miles":              round(total_distance, 2),
        "total_fuel_gallons":                total_gallons,
        "station_refuel_cost_usd":           station_cost,
        "trip_consumption_cost_usd":         station_cost,   # no double-count
        "total_fuel_cost":                   station_cost,   # backward-compat alias
        "fuel_remaining_at_destination_gal": round(fuel_remaining, 2),
        "number_of_stops":                   len(stops_chosen),
        "reachable":                         True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_fuel_route(start_query: str, finish_query: str) -> dict:
    """
    End-to-end route planning pipeline.

    1. Geocode origin and destination (offline-first, Nominatim fallback).
    2. Fetch driving route via OSRM (single network call, cached per session).
    3. Identify fuel stop candidates within ``CORRIDOR_RADIUS_MILES`` of route.
    4. Select globally optimal stops via DAG Dynamic Programming.

    Args:
        start_query:  Human-readable origin string (e.g. ``"Chicago, IL"``).
        finish_query: Human-readable destination string (e.g. ``"Austin, TX"``).

    Returns:
        A dict matching the ``RouteResponseSerializer`` schema.

    Raises:
        LocationNotFoundError: if either location cannot be geocoded.
        RoutingServiceError:   if the OSRM call fails.
    """
    lat1, lon1, start_name  = geocode_location(start_query)
    lat2, lon2, finish_name = geocode_location(finish_query)

    total_dist, route_pts, polyline_str = fetch_driving_route((lat1, lon1), (lat2, lon2))

    candidates = find_corridor_stops(route_pts, total_dist)
    plan = optimise_refuel_stops(candidates, total_dist)

    return {
        "start":         start_name,
        "finish":        finish_name,
        "start_coords":  {"lat": lat1, "lon": lon1},
        "finish_coords": {"lat": lat2, "lon": lon2},
        **plan,
        "route_geometry": polyline_str,
    }
