from mlopenapp.forms import UploadFileForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView


class DataView(TemplateView, FormView):
    template_name = "base.html"
    form_class = UploadFileForm
    success_url = '/data/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "MLopen"