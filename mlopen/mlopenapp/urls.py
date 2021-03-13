from django.urls import path
from django.conf.urls import url

from . import views

app_name = "mlopenapp"
urlpatterns = [
    url(
        r'^$',
        views.IndexView.as_view(),
        name="base"
    ),
]
