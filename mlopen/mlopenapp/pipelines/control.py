import os
import csv
import pandas as pd
from text_preprocessing import process_df
from input.text_files_input import prepare_data


def prepare_data():
    df = prepare_data()
    df = process_df(df, 'text')
    print(df)


prepare_data()