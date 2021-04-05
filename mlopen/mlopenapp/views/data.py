import os
from mlopenapp.forms import UploadFileForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

from ..pipelines import control, text_preprocessing as tpp
from ..pipelines.input import text_files_input as tfi
from ..utils import io_handler as io


class DataView(TemplateView, FormView):
    template_name = "data.html"
    form_class = UploadFileForm
    success_url = '/data/'
    relative_url = "data"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Data"
        context['template'] = "data.html"
        return context
