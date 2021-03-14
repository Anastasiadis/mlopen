import os
import csv
import pandas as pd

def prepare_data():
    df1 = read_from_dir(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                     'data/user_data/train/pos/'), 1)
    df2 = read_from_dir(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                     'data/user_data/train/neg/'), 0)
    df = pd.concat([df1, df2], ignore_index=True)
    #print(df)
    return(df)

def read_from_dir(dir, sentiment=None):
    data = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            text = f.read()
            if sentiment is not None:
                data.append([text, sentiment])
            else:
                data.append(text)
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    #print(df)
    return df


def read_multiple_to_csv(dir, sentiment=None):
    _csv = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            text = f.read()
            if sentiment is not None:
                _csv.append([text, sentiment])
            else:
                _csv.append(text)
    print(dir.split(os.path.sep)[-2])
    print(dir)
    print((os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                        os.pardir, 'data', 'csvs', dir.split(os.path.sep)[-2
                        ] + '.csv')))
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                           os.pardir, 'data', 'csvs', dir.split(os.path.sep)[-2
                                                      ] + '.csv'), 'w') as out:
        csv_out = csv.writer(out)
        for row in _csv:
            print(type(row))
            csv_out.writerow(row)


def get_from_csvs(dir=None, csvs=None):
    if dir is not None:
        df = None
        for filename in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, filename)) if df is None else pd.concat(
                [df, pd.read_csv(os.path.join(dir, filename))], ignore_index=True)
        return df
    elif type(csvs) is list:
        df = pd.read_csv(csvs[0])
        for cs in csvs[1:]:
            df = pd.concat([df, pd.read_csv(cs)], ignore_index=True)
        return df



#print(os.path.join(os.pardir, 'data/user_data/train/pos/'))
#read_from_dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data/user_data/train/pos/'), 1)
#read_from_dir(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data/user_data/train/neg/'), 0)
#prepare_data()