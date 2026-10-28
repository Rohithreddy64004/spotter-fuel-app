"""
fuel_route/serializers.py

DRF serializers for the Spotter Fuel Route Optimizer.

Request validation
------------------
``RouteRequestSerializer`` strips whitespace and rejects identical origin /
destination pairs before the service layer is ever invoked.

Response schema
---------------
``RouteResponseSerializer`` documents the disambiguated fuel accounting fields:

    station_refuel_cost_usd
        Actual dollars spent at the pump at each selected intermediate stop.
        Zero for trips that fit within one full tank.

    trip_consumption_cost_usd
        Economic cost of all fuel consumed over the entire trip.
        Equals ``station_refuel_cost_usd`` under the Full-Tank-to-Next-Waypoint
        model (origin leg is pre-paid; last stop already purchased fuel for the
        terminal leg; no separate final-leg cost is added).

    total_fuel_cost
        Backward-compatible alias. Always equals ``trip_consumption_cost_usd``.

    fuel_remaining_at_destination_gal
        Gallons left in the tank on arrival (0.0 when each stop purchases
        exactly enough for the next leg; positive for short trips that arrive
        with leftover fuel from the initial fill).
"""

from rest_framework import serializers


class RouteRequestSerializer(serializers.Serializer):
    start = serializers.CharField(
        max_length=200,
        required=True,
        error_messages={"required": "An origin location is required."},
    )
    finish = serializers.CharField(
        max_length=200,
        required=True,
        error_messages={"required": "A destination location is required."},
    )

    def validate_start(self, value: str) -> str:
        return value.strip()

    def validate_finish(self, value: str) -> str:
        return value.strip()

    def validate(self, attrs: dict) -> dict:
        if attrs.get("start", "").lower() == attrs.get("finish", "").lower():
            raise serializers.ValidationError(
                "Origin and destination must be different locations."
            )
        return attrs


class CoordinateSerializer(serializers.Serializer):
    lat = serializers.FloatField()
    lon = serializers.FloatField()


class FuelStopSerializer(serializers.Serializer):
    opis_id            = serializers.IntegerField()
    name               = serializers.CharField()
    address            = serializers.CharField()
    city               = serializers.CharField()
    state              = serializers.CharField(max_length=10)
    retail_price       = serializers.FloatField()
    latitude           = serializers.FloatField()
    longitude          = serializers.FloatField()
    mile_along_route   = serializers.FloatField()
    leg_distance_miles = serializers.FloatField()
    gallons_pumped     = serializers.FloatField()
    cost               = serializers.FloatField()


class RouteResponseSerializer(serializers.Serializer):
    start         = serializers.CharField()
    finish        = serializers.CharField()
    start_coords  = CoordinateSerializer()
    finish_coords = CoordinateSerializer()

    total_distance_miles = serializers.FloatField()
    total_fuel_gallons   = serializers.FloatField()

    # ── Disambiguated fuel accounting ──────────────────────────────────────
    station_refuel_cost_usd          = serializers.FloatField()
    trip_consumption_cost_usd        = serializers.FloatField()
    total_fuel_cost                  = serializers.FloatField()   # alias
    fuel_remaining_at_destination_gal = serializers.FloatField()

    number_of_stops = serializers.IntegerField(min_value=0)
    fuel_stops      = FuelStopSerializer(many=True)
    route_geometry  = serializers.CharField()
    reachable       = serializers.BooleanField(default=True)
