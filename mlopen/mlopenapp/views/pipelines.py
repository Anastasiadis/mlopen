import os
import importlib.util

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
        print("SNIIIIIIIIIIIFFF")
        print(form)
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            print("AT LEAST IT'S AJAX")
            if "pipelines" in clean_data:
                if clean_data['pipelines'] == 'control':
                    return self.update(clean_data)
                else:
                    return None
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def form_valid(self, form):
        print("HURRAAAAYYYYYYYYY")
        print(form)
        if self.request.is_ajax():
            return self.update(form)
        return self.render_to_response(self.get_context_data(form=form))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Pipelines"
        context['template'] = "pipelines.html"
        print(context)

        return context

    def update(self, clean_data):
        pipeline = clean_data['pipelines']
        """

               list = io.get_pipeline_list()
               context['pipeline_list'] = list
               print(list)

               # These will be replaced by user input
               train_paths = [
                   os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                                'data/user_data/train/pos/'),
                   os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                                'data/user_data/train/neg/')
               ]

               train_sentiments = [1, 0]

               test_paths = [
                   os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                                'data/user_data/test/pos/'),
                   os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                                'data/user_data/test/neg/')
               ]

               test_sentiments = [1, 0]
               df_train = tpp.process_text_df(tfi.prepare_data(train_paths, train_sentiments),
                                              'text')
               df_test = tpp.process_text_df(tfi.prepare_data(test_paths, test_sentiments),
                                             'text')
               """

        spec = importlib.util.spec_from_file_location(pipeline,
                                                      os.path.join(constants.CONTROL_DIR,
                                                                   'control.py'))
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
        preds = control.make_prediction(
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
        ret = {'data': preds, 'columns': ["Original Statement", "Predicted Sentiment"]}
        print(ret)

        # """
        return JsonResponse(ret, safe=False)

