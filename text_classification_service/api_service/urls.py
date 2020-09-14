
from django.urls import path
from .views import TextClassificationView

app_name = "api_service"

urlpatterns = [
    path('', TextClassificationView.as_view(), name="api_service"),

]

