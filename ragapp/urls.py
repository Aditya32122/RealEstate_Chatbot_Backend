from django.urls import path
from .views import upload_csv, query_view, check_data,health_check

urlpatterns = [
    path("upload-csv", upload_csv),
    path("query", query_view),
    path("check-data", check_data),
    path("health-check", health_check),
]