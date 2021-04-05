from django import forms
from .utils import io_handler as io


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()


class PipelineSelectForm(forms.Form):
    pipelines = forms.ChoiceField()
    input = forms.ChoiceField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline_list = [(o, o) for i, o in enumerate(io.get_pipeline_list())]
        self.fields['pipelines'] = forms.ChoiceField(
            choices=pipeline_list
        )
