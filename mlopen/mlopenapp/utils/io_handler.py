import sys
import os
import pickle
import datetime
from django.db import models
from django.core.files import File
from .. import constants


def save(arg_object, name, save_to_db=False, type=None):
    try:
        output = open(name + '.pkl', 'wb')
        pickle.dump(arg_object, output, pickle.HIGHEST_PROTOCOL)
        output.close()
        if save_to_db:
            output = open(name + '.pkl', 'rb')
            filefield = constants.FILE_TYPES[type](
                name=name,
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                file=File(output))
            filefield.save()
            return filefield
        return True
    except AttributeError as e:
        print("EXCEPTION IS ")
        print(e)
        return False


def load(name, type):
    try:
        with open(os.path.join(constants.FILE_DIRS[type], name), 'rb') as input:
            ret = pickle.load(input)
        return ret
    except Exception as e:
        print(e)
        return False


def save_pipeline(models, args, name):
    pip_models = []
    pip_args = []
    for model in models:
         temp = save(model[0], model[1], True, 'model')
         if type(temp) == bool:
            return False
         pip_models.append(temp)
    for arg in args:
         temp = save(arg[0], arg[1], True, 'arg')
         if type(temp) == bool:
            return False
         pip_args.append(temp)
    pipeline = constants.FILE_TYPES['pipeline'](
        name=name,
        control=name,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now())
    pipeline.save()
    for model in pip_models:
        pipeline.ml_models.add(model)
    for arg in pip_args:
        pipeline.ml_args.add(arg)
    pipeline.save()


def get_pipeline_list():
    pipeline_list = []
    for filename in os.listdir(constants.CONTROL_DIR):
        if filename.endswith("control.py"):
            pipeline_list.append(filename[:-3])
    return pipeline_list


def save_pipeline_file(f):
    with open(os.path.join(constants.CONTROL_DIR, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
