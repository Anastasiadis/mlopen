import os
import importlib.util
import pandas

from django import forms
from mlopenapp.forms import PipelineSelectForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.http import JsonResponse
from ..models import MLPipeline as Pipeline
from ..pipelines import text_preprocessing as tpp
from ..pipelines.input import text_files_input as tfi

from ..utils import io_handler as io

from .. import constants
from ..utils import params_handler


class PipelineView(TemplateView, FormView):
    template_name = "pipelines.html"
    form_class = PipelineSelectForm
    success_url = '/pipelines/'
    relative_url = "pipelines"
    CHOICES = [(0, 'Run Pipeline'),
               (1, 'Train Model')]

    def get_form(self, form_class=PipelineSelectForm):
        form = super().get_form(form_class)
        form.fields["type"] = forms.ChoiceField(choices=self.CHOICES, initial=0, required=False)
        return form

    def form_invalid(self, form):
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            data = self.request.POST.get("select_pipeline", False)
            if data:
                pipeline = self.request.POST.get("pipeline", False)
                pipeline = Pipeline.objects.filter(id=int(pipeline)).first()
                spec = importlib.util.spec_from_file_location(pipeline.control,
                                                              os.path.join(
                                                                  constants.CONTROL_DIR,
                                                                  str(pipeline.control) + '.py'))
                control = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(control)
                params = control.get_params()
                userform = params_handler.get_params_form(params)
                return self.update_attrs(userform.as_table())
            if "pipelines" in clean_data:
                return self.update(clean_data)
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def form_valid(self, form):
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            data = self.request.POST.get("select_pipeline", False)
            if data:
                userform = forms.Form()
                userform.fields["sth"] = forms.CharField()
                return self.render_to_response(
                    self.get_context_data(form=form, userform="oooo"))
            if "pipelines" in clean_data:
                return self.update(clean_data)
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Pipelines"
        context['template'] = "pipelines.html"
        if "userform" in kwargs:
            context["userform"] = kwargs["userform"]
        return context

    def update_attrs(self, userform):
        ret = {"userform": userform}
        return JsonResponse(ret, safe=False)

    def update(self, clean_data):
        inpt = [
                "This was a very good movie indeed, I enjoyed it very much!",
                "A very bad movie, awful visuals, horrible sound - I hated it.",
                "I was sceptical at first, but this movie won me over - a great documentary!",
                "I would never watch this a second time, it was mediocre at best.",
                "Who would have thought that such an expensive play would be so low quality.",
                "Please, don't watch this! It's a total waste of time!",
                "I thought I would not like this, but it turned out to be pretty good!",
                "I would wait to rent this. It does not justify a full price ticket.",
                "If you have one movie to watch, then watch this! You'll be left in awe!",
                "Started good, but it became too slow and unimaginative in the end.",
            ]
        pipeline = clean_data['pipelines']
        spec = importlib.util.spec_from_file_location(pipeline.control,
                                                      os.path.join(constants.CONTROL_DIR,
                                                                   str(pipeline.control) + '.py'))
        control = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control)

        """params = control.get_params()
        for field, _ in params.items():
            print(field)"""

        params = dict(self.request.POST)
        for name in ['type', 'pipelines', 'input']:
            params.pop(name, None)
        for name, param in params.items():
            if isinstance(param, list) and len(param) == 1:
                params[name] = param[0]

        if clean_data["type"] == "0":
            model = None
            args = {}
            pipeline_ret = io.load_pipeline(pipeline)
            if pipeline_ret:
                model = pipeline_ret[0]
                args = pipeline_ret[1]
            preds = control.run_pipeline(inpt, model, args, params)
            ret = {'data': preds['data'], 'columns': preds['columns'], 'graphs': preds['graphs']}
        else:
            ret = None

        return JsonResponse(ret, safe=False)

