from django.urls import path
from .views import upload_csv, query_view, check_data

urlpatterns = [
    path("upload-csv", upload_csv),
    path("query", query_view),
    path("check-data", check_data),
]