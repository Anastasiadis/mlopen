import os
import importlib.util
import pandas

from mlopenapp.forms import PipelineSelectForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.http import JsonResponse
from ..pipelines import text_preprocessing as tpp
from ..pipelines.input import text_files_input as tfi

from ..utils import io_handler as io

from .. import constants


class PipelineView(TemplateView, FormView):
    template_name = "pipelines.html"
    form_class = PipelineSelectForm
    success_url = '/pipelines/'
    relative_url = "pipelines"

    def form_invalid(self, form):
        print(form)
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            print("AT LEAST IT'S AJAX")
            if "pipelines" in clean_data:
                return self.update(clean_data)
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def form_valid(self, form):
        print("INVALID")
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            print("AT LEAST IT'S AJAX")
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
        return context

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
        df_inpt = pandas.DataFrame(inpt, columns=['text'])

        pipeline = clean_data['pipelines']
        print("NAME IS")
        print(pipeline.control)
        if (pipeline.control == "ltsm_sa_control.py"):
            spec = importlib.util.spec_from_file_location('ltsm_sa_control',
                                                          os.path.join(constants.CONTROL_DIR,
                                                                       str(
                                                                           'ltsm_sa_control.py')))
            control = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(control)

            model = io.load("lstm_model.pkl", 'model')
            args = {}
            args['lstm_vocab'] = io.load("lstm_vocab.pkl", 'arg')
            args['lstm_words'] = io.load("lstm_words.pkl", 'arg')

            preds = control.run_pipeline(df_inpt, 'text', model, args)

        else:
            print(pipeline.control)
            print(os.path.join(constants.CONTROL_DIR, 'control.py'))
            spec = importlib.util.spec_from_file_location(pipeline.control,
                                                          os.path.join(constants.CONTROL_DIR,
                                                                       str(pipeline.control) + '.py'))
            control = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(control)

            # control.train(df_train, df_test, 'text')

            # """
            model = io.load("logreg_model.pkl", 'model')
            tfidf = io.load("tfidf_vect.pkl", 'arg')
            print(tfidf)
            print(model)
            print(type(tfidf))
            print(type(model))
            print("HAAAAAAAOOOOOOO")
            preds = control.run_pipeline(
                [
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
                , tfidf, model)
        ret = {'data': preds['data'], 'columns': ["Original Statement", "Predicted Sentiment"], 'graphs': preds['graphs']}
        print(ret)

        # """
        return JsonResponse(ret, safe=False)

