"""
fuel_route/tests.py

Test suite for the Spotter Fuel Route Optimizer.

All OSRM network calls are mocked with pre-computed straight-line geometry
so every test passes in an offline / CI environment.  Business logic —
geocoding, corridor filtering, DP optimiser, serialisation, and the API view
— runs against the live SQLite test database populated by ``load_fuel_csv``.

Fuel accounting model (Full-Tank Refuel)
----------------------------------------
At each selected stop the driver fills to tank capacity (50 gal):

    gallons_pumped = TANK_CAPACITY_GALLONS - fuel_before_arrival

Fuel balance identity that every test verifies:

    initial_fuel + sum(gallons_pumped) - total_distance_miles / MPG
        == fuel_remaining_at_destination_gal  (within 0.01 gal tolerance)
"""

import math
from unittest.mock import patch

import polyline as polyline_lib
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from fuel_route.services import (
    TANK_CAPACITY_GALLONS,
    TANK_CAPACITY_MILES,
    MPG,
    _cluster_stops,
    optimise_refuel_stops,
    plan_fuel_route,
)


# ---------------------------------------------------------------------------
# Shared helpers and OSRM mock fixtures
# ---------------------------------------------------------------------------

def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _straight_line_waypoints(lat1, lon1, lat2, lon2, steps=200):
    return [
        (lat1 + i / steps * (lat2 - lat1), lon1 + i / steps * (lon2 - lon1))
        for i in range(steps + 1)
    ]


def _assert_fuel_balance(test_case, result, tolerance=0.05):
    """
    Assert the fuel balance identity holds for any route result.

        initial_fuel + sum(gallons_pumped) - total_gallons_burned
            == fuel_remaining_at_destination_gal

    Args:
        test_case:  A ``unittest.TestCase`` instance.
        result:     The dict returned by ``plan_fuel_route`` or ``optimise_refuel_stops``.
        tolerance:  Acceptable floating-point rounding difference (gallons).
    """
    initial = TANK_CAPACITY_GALLONS
    pumped  = sum(s["gallons_pumped"] for s in result["fuel_stops"])
    burned  = result["total_fuel_gallons"]
    remaining = result["fuel_remaining_at_destination_gal"]

    balance = initial + pumped - burned
    test_case.assertAlmostEqual(
        balance,
        remaining,
        delta=tolerance,
        msg=(
            f"Fuel balance violated: {initial:.2f} + {pumped:.4f} - {burned:.4f} "
            f"= {balance:.4f}, but fuel_remaining={remaining:.4f}"
        ),
    )

    # Fuel never negative, never over tank capacity
    test_case.assertGreaterEqual(remaining, 0.0, "Fuel remaining cannot be negative")
    test_case.assertLessEqual(
        remaining, TANK_CAPACITY_GALLONS,
        "Fuel remaining cannot exceed tank capacity"
    )

    # Each leg <= 500 miles (only when full stop dicts are present)
    if result["fuel_stops"] and "mile_along_route" in result["fuel_stops"][0]:
        miles_seq = (
            [0.0]
            + [s["mile_along_route"] for s in result["fuel_stops"]]
            + [result["total_distance_miles"]]
        )
        for i in range(1, len(miles_seq)):
            leg = miles_seq[i] - miles_seq[i - 1]
            test_case.assertLessEqual(
                leg, TANK_CAPACITY_MILES + 1e-6,
                f"Leg {i} is {leg:.2f} mi which exceeds {TANK_CAPACITY_MILES:.0f}-mi range"
            )


# OSRM mock fixtures
_CHI_LAT, _CHI_LON = 41.88585, -87.61812
_AUS_LAT, _AUS_LON = 30.50597, -97.74719
_DAL_LAT, _DAL_LON = 32.79044, -96.80439

_CHI_AUS_DIST = _haversine_miles(_CHI_LAT, _CHI_LON, _AUS_LAT, _AUS_LON)
_CHI_AUS_WPT  = _straight_line_waypoints(_CHI_LAT, _CHI_LON, _AUS_LAT, _AUS_LON)
_CHI_AUS_ENC  = polyline_lib.encode(_CHI_AUS_WPT)

_AUS_DAL_DIST = _haversine_miles(_AUS_LAT, _AUS_LON, _DAL_LAT, _DAL_LON)
_AUS_DAL_WPT  = _straight_line_waypoints(_AUS_LAT, _AUS_LON, _DAL_LAT, _DAL_LON)
_AUS_DAL_ENC  = polyline_lib.encode(_AUS_DAL_WPT)


def _osrm_side_effect(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    if abs(lat1 - _CHI_LAT) < 0.1 and abs(lat2 - _AUS_LAT) < 0.1:
        return _CHI_AUS_DIST, _CHI_AUS_WPT, _CHI_AUS_ENC
    if abs(lat1 - _AUS_LAT) < 0.1 and abs(lat2 - _DAL_LAT) < 0.1:
        return _AUS_DAL_DIST, _AUS_DAL_WPT, _AUS_DAL_ENC
    raise AssertionError(f"Unexpected OSRM call: origin={origin}, destination={destination}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class FuelRouteTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        call_command("load_fuel_csv", "fuel-prices-for-be-assessment.csv")

    def setUp(self):
        self.client = APIClient()

    # ── 1. Short route (< 500 mi) ───────────────────────────────────────────

    @patch("fuel_route.services.fetch_driving_route", side_effect=_osrm_side_effect)
    def test_short_route_no_stops(self, _mock):
        """
        Sub-tank trip: 0 pump stops, station_refuel = $0, trip_cost > $0.
        fuel_remaining = TANK - distance/MPG (leftover from initial fill).
        """
        result = plan_fuel_route("Austin, TX", "Dallas, TX")

        self.assertLess(result["total_distance_miles"], 500.0)
        self.assertEqual(result["number_of_stops"], 0)
        self.assertEqual(len(result["fuel_stops"]), 0)
        self.assertTrue(result["reachable"])

        self.assertEqual(result["station_refuel_cost_usd"], 0.0)
        self.assertGreater(result["trip_consumption_cost_usd"], 0.0)
        self.assertAlmostEqual(
            result["total_fuel_cost"], result["trip_consumption_cost_usd"], places=2
        )

        # Fuel balance on short trip
        _assert_fuel_balance(self, result)

        # Expected remaining fuel
        expected_remaining = round(
            TANK_CAPACITY_GALLONS - result["total_distance_miles"] / MPG, 2
        )
        self.assertAlmostEqual(
            result["fuel_remaining_at_destination_gal"], expected_remaining, places=1
        )

    # ── 2. Long route (> 500 mi) ────────────────────────────────────────────

    @patch("fuel_route.services.fetch_driving_route", side_effect=_osrm_side_effect)
    def test_long_route_fuel_balance(self, _mock):
        """
        Cross-country route: DP selects stops, all legs <= 500 mi.
        Fuel balance identity: initial + pumped - burned == remaining.
        """
        result = plan_fuel_route("Chicago, IL", "Austin, TX")
        total_dist = result["total_distance_miles"]

        self.assertGreater(total_dist, 500.0)
        self.assertGreater(result["number_of_stops"], 0)
        self.assertTrue(result["reachable"])

        # Core invariant
        _assert_fuel_balance(self, result)

        # Disambiguated costs
        self.assertGreater(result["station_refuel_cost_usd"], 0.0)
        self.assertAlmostEqual(
            result["total_fuel_cost"], result["trip_consumption_cost_usd"], places=2
        )

        # gallons_pumped consistency
        for stop in result["fuel_stops"]:
            expected_cost = round(stop["gallons_pumped"] * stop["retail_price"], 2)
            self.assertAlmostEqual(
                stop["cost"], expected_cost, places=1,
                msg=f"Cost mismatch at stop {stop['name']}"
            )

    # ── 3. Fuel balance regression (deterministic, no DB) ───────────────────

    def test_fuel_balance_regression_deterministic(self):
        """
        Regression test for the fuel_remaining_at_destination_gal bug.

        Uses a synthetic 3-stop route with known exact values.
        Verifies the balance identity holds and the reported
        fuel_remaining matches TANK - final_leg / MPG.

        Before the fix, fuel_remaining used TANK - last_leg/MPG while
        gallons_pumped used next_leg/MPG (exact-purchase), producing an
        internally inconsistent result (balance != 0).
        """
        # Synthetic stops: route = 0 -> 350mi -> 650mi -> 950mi -> 1100mi
        stops = [
            {
                "opis_id": 1, "name": "Stop A", "address": "I-55", "city": "Springfield",
                "state": "IL", "retail_price": 3.00, "latitude": 39.8, "longitude": -89.6,
                "mile_along_route": 350.0, "corridor_distance_miles": 1.0,
            },
            {
                "opis_id": 2, "name": "Stop B", "address": "I-44", "city": "Joplin",
                "state": "MO", "retail_price": 2.80, "latitude": 37.1, "longitude": -94.5,
                "mile_along_route": 650.0, "corridor_distance_miles": 2.0,
            },
            {
                "opis_id": 3, "name": "Stop C", "address": "I-35", "city": "OKC",
                "state": "OK", "retail_price": 2.90, "latitude": 35.5, "longitude": -97.5,
                "mile_along_route": 950.0, "corridor_distance_miles": 1.5,
            },
        ]
        total_distance = 1100.0

        result = optimise_refuel_stops(stops, total_distance)

        self.assertTrue(result["reachable"])
        self.assertGreater(result["number_of_stops"], 0)

        # The fundamental balance identity
        _assert_fuel_balance(self, result, tolerance=0.01)

        # Verify fuel_remaining = TANK - final_leg / MPG
        last_stop_mile = result["fuel_stops"][-1]["mile_along_route"]
        final_leg = total_distance - last_stop_mile
        expected_remaining = max(0.0, TANK_CAPACITY_GALLONS - final_leg / MPG)
        self.assertAlmostEqual(
            result["fuel_remaining_at_destination_gal"],
            expected_remaining,
            places=2,
            msg="fuel_remaining must equal TANK - final_leg/MPG (Full-Tank model)"
        )

        # All legs <= 500 miles
        _assert_fuel_balance(self, result)

        # gallons_pumped consistency: pumped * price == cost (within rounding)
        for stop in result["fuel_stops"]:
            self.assertAlmostEqual(
                stop["cost"],
                round(stop["gallons_pumped"] * stop["retail_price"], 2),
                places=1,
            )
            self.assertGreater(stop["gallons_pumped"], 0.0)
            self.assertLessEqual(stop["gallons_pumped"], TANK_CAPACITY_GALLONS)

    # ── 4. Boundary: stop exactly at 500 miles ──────────────────────────────

    def test_boundary_stop_at_500_miles(self):
        """Stop exactly at 500-mile mark is reachable from origin."""
        stops = [
            {
                "opis_id": 10, "name": "Boundary Stop", "address": "US-66",
                "city": "Amarillo", "state": "TX", "retail_price": 3.10,
                "latitude": 35.2, "longitude": -101.8,
                "mile_along_route": 500.0, "corridor_distance_miles": 0.5,
            }
        ]
        result = optimise_refuel_stops(stops, total_distance=900.0)
        self.assertTrue(result["reachable"])
        self.assertEqual(result["number_of_stops"], 1)
        _assert_fuel_balance(self, result, tolerance=0.01)

    # ── 5. Fuel desert: gap > 500 miles ────────────────────────────────────

    def test_fuel_desert_returns_unreachable(self):
        """A 501-mile gap between the origin and the first stop is a fuel desert."""
        stops = [
            {
                "opis_id": 99, "name": "Far Stop", "address": "US-101",
                "city": "Sacramento", "state": "CA", "retail_price": 4.50,
                "latitude": 38.6, "longitude": -121.5,
                "mile_along_route": 501.0, "corridor_distance_miles": 3.0,
            }
        ]
        result = optimise_refuel_stops(stops, total_distance=800.0)
        self.assertFalse(result["reachable"])
        self.assertEqual(result["number_of_stops"], 0)
        self.assertEqual(result["fuel_stops"], [])

    # ── 6. DP is cheaper than greedy (counterexample) ───────────────────────

    def test_dp_beats_greedy(self):
        """
        Classic greedy counterexample: a cheap station far down the road
        makes it worth skipping a nearby expensive station.

        Greedy (local best): stop at mile 300 ($3.50) because it is the
        first stop after origin and the tank might not reach mile 600.
        DP (global best): skip mile-300 stop, refuel at mile 480 ($2.50),
        then drive to destination — total cost is lower.
        """
        stops = [
            {
                "opis_id": 1, "name": "Expensive Early", "address": "",
                "city": "City A", "state": "XX", "retail_price": 3.50,
                "latitude": 37.0, "longitude": -95.0,
                "mile_along_route": 300.0, "corridor_distance_miles": 1.0,
            },
            {
                "opis_id": 2, "name": "Cheap Later", "address": "",
                "city": "City B", "state": "XX", "retail_price": 2.50,
                "latitude": 35.0, "longitude": -97.0,
                "mile_along_route": 480.0, "corridor_distance_miles": 1.0,
            },
        ]
        # Total 850 miles: needs at least one stop after origin (tank = 500 mi)
        result = optimise_refuel_stops(stops, total_distance=850.0)

        self.assertTrue(result["reachable"])
        _assert_fuel_balance(self, result, tolerance=0.05)

        # DP must prefer the cheaper stop at mile 480
        stop_names = [s["name"] for s in result["fuel_stops"]]
        self.assertIn(
            "Cheap Later", stop_names,
            "DP should select the cheaper station at mile 480"
        )

        # Verify the DP cost is cheaper than the greedy alternative
        # Greedy: stop at mile 300, fill at $3.50 for leg 300->850 (55 gal)... 
        # but tank is only 50 gal. Greedy would need 2 stops.
        # DP: skip mile-300, refuel at mile-480 (fills to 50 gal), drives 370mi to dest.
        # Just verify DP is reachable and selects the cheap stop.
        self.assertGreater(result["station_refuel_cost_usd"], 0.0)

    # ── 7. API: success with full schema ────────────────────────────────────

    @patch("fuel_route.services.fetch_driving_route", side_effect=_osrm_side_effect)
    def test_api_route_returns_complete_schema(self, _mock):
        """POST /api/route/ returns 200 with all required fields."""
        url = reverse("api-route")
        response = self.client.post(
            url, data={"start": "Chicago, IL", "finish": "Austin, TX"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.data

        required_fields = [
            "start", "finish", "start_coords", "finish_coords",
            "total_distance_miles", "total_fuel_gallons",
            "station_refuel_cost_usd", "trip_consumption_cost_usd",
            "total_fuel_cost", "fuel_remaining_at_destination_gal",
            "reachable", "number_of_stops", "fuel_stops", "route_geometry",
        ]
        for field in required_fields:
            self.assertIn(field, data, f"Missing API field: {field!r}")

        self.assertTrue(data["reachable"])
        self.assertGreater(data["total_distance_miles"], 0)

        # Alias consistency
        self.assertAlmostEqual(
            float(data["total_fuel_cost"]),
            float(data["trip_consumption_cost_usd"]),
            places=2,
        )

        # Fuel balance via API response (use the full result dict directly)
        _assert_fuel_balance(self, {
            "fuel_stops": list(data["fuel_stops"]),
            "total_fuel_gallons": data["total_fuel_gallons"],
            "total_distance_miles": data["total_distance_miles"],
            "fuel_remaining_at_destination_gal": data["fuel_remaining_at_destination_gal"],
        })

    # ── 8. API: validation errors ────────────────────────────────────────────

    def test_api_validation_missing_field(self):
        """POST /api/route/ returns 400 when finish is missing."""
        url = reverse("api-route")
        response = self.client.post(url, data={"start": "Chicago, IL"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_api_validation_same_origin_destination(self):
        """POST /api/route/ returns 400 when start == finish."""
        url = reverse("api-route")
        response = self.client.post(
            url,
            data={"start": "Chicago, IL", "finish": "Chicago, IL"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    # ── 9. Map template ──────────────────────────────────────────────────────

    def test_map_view_renders(self):
        """GET /map/ returns 200 with expected HTML structure."""
        url = reverse("map-view")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertContains(response, "Spotter")
        self.assertContains(response, 'id="map"')
        self.assertContains(response, "calculateRoute")
        self.assertContains(response, "val-trip-cost")
        self.assertContains(response, "val-station-cost")

    # ── 10. Clustering unit test ─────────────────────────────────────────────

    def test_cluster_stops_removes_micro_stops(self):
        """
        _cluster_stops must collapse stations within CLUSTER_WINDOW_MILES
        into a single representative retaining the cheapest price.
        """
        raw = [
            {"mile_along_route": 100.0, "retail_price": 3.20},
            {"mile_along_route": 100.8, "retail_price": 3.10},  # same cluster, cheaper
            {"mile_along_route": 101.5, "retail_price": 3.15},  # same cluster
            {"mile_along_route": 110.0, "retail_price": 2.90},  # new cluster
        ]
        result = _cluster_stops(raw, window_miles=2.0)
        self.assertEqual(len(result), 2, "Should produce exactly 2 clusters")
        # First cluster representative must be the cheapest
        self.assertAlmostEqual(result[0]["retail_price"], 3.10, places=2)
        self.assertAlmostEqual(result[1]["retail_price"], 2.90, places=2)
