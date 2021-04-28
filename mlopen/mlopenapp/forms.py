from django import forms
from .utils import io_handler as io
from . import models


class UploadFileForm(forms.Form):
    name = forms.CharField(max_length=50)
    file = forms.FileField()


class PipelineSelectForm(forms.Form):
    pipelines = forms.ModelChoiceField(queryset=models.MLPipeline.objects.all())
    input = forms.ModelChoiceField(queryset=models.InputFile.objects.all())


class UploadForm(forms.ModelForm):
    class Meta:
        model = models.InputFile
        fields = [
        'name',
        'created_at',
        'file'
        ]
