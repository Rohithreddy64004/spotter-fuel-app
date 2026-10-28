import logging

from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from fuel_route.serializers import RouteRequestSerializer, RouteResponseSerializer
from fuel_route.services import (
    LocationNotFoundError,
    RoutingServiceError,
    plan_fuel_route,
)

logger = logging.getLogger(__name__)


class RouteAPIView(APIView):
    """
    POST /api/route/

    Accepts an origin and destination, returns an optimised driving route
    with recommended refuelling stops, per-stop costs, and a encoded polyline.
    """

    def post(self, request, *args, **kwargs):
        serializer = RouteRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Invalid request.", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        start = serializer.validated_data["start"]
        finish = serializer.validated_data["finish"]

        try:
            result = plan_fuel_route(start, finish)
        except LocationNotFoundError as exc:
            logger.warning("Location lookup failed [%s → %s]: %s", start, finish, exc)
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RoutingServiceError as exc:
            logger.error("Routing service error [%s → %s]: %s", start, finish, exc)
            return Response(
                {"error": "The routing service is temporarily unavailable. Please try again."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.exception("Unexpected error computing route [%s → %s]", start, finish)
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        out = RouteResponseSerializer(data=result)
        if out.is_valid():
            return Response(out.data, status=status.HTTP_200_OK)

        # Serializer validation failure is a server-side contract bug — log it
        logger.error("RouteResponseSerializer rejected service output: %s", out.errors)
        return Response(result, status=status.HTTP_200_OK)


class MapView(TemplateView):
    """Serves the interactive Leaflet.js route planner UI at /map/."""

    template_name = "fuel_route/map.html"
