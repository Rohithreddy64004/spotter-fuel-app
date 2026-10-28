from django.urls import path
from fuel_route.views import RouteAPIView, MapView

urlpatterns = [
    path("api/route/", RouteAPIView.as_view(), name="api-route"),
    path("map/", MapView.as_view(), name="map-view"),
]

