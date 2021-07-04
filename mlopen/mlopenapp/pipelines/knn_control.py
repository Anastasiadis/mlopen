import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from mlopenapp.utils import io_handler as io


def get_params(run=True):
    if run:
        params = {"k": ("integer", {"default": 5}), "data": ("file"), "column_names": ("string")}
        return params


def train():
    io.save_pipeline([], [], os.path.basename(__file__))


def run_pipeline(input, model, args, params=None):
    input.open("r")
    # Assign colum names to the dataset
    if "column_names" in params and len(params['column_names']) > 0:
        names = params['column_names'].split(",")
    else:
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(input, names=names)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Normalize attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=int(params.get('k', 5)))
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(y_pred)
    preds = {}
    print(dataset.columns.values[:-1])
    data = [[str(x[0]), str(x[1]), str(x[2]), str(x[3]), y, z] for x, y, z in zip(X_test, y_pred, y_test)]
    preds['data'] = data
    preds['columns'] = list(dataset.columns.values)[:-1] + ['Predicted Class', 'Actual Class']
    preds['graphs'] = None
    print(preds['data'])
    print(preds['columns'])
    return preds

