from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("favicon.ico", lambda req: HttpResponse(status=204)),
    path("", include("fuel_route.urls")),
    path("", RedirectView.as_view(url="/map/", permanent=False)),
]
