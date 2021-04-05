import os
from . import models
from django.conf import settings
from .models import storages

FILE_TYPES = {
    "model": models.MLModel,
    "arg": models.MLArgs,
    "pipeline": models.MLPipeline
}

FILE_DIRS = {
    "model": os.path.join(storages.LOCAL_STORAGE, 'models'),
    "arg": os.path.join(storages.LOCAL_STORAGE, 'args'),
    "pipeline": os.path.join(storages.LOCAL_STORAGE, 'pipelines')
}

CONTROL_DIR = os.path.join(settings.BASE_DIR, 'mlopenapp/pipelines')
