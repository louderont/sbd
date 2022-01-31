import os
from typing import Union
import pickle
import pandas as pd
import pathlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from sbdetection.preprocessing.preprocessor import binary_target_to_sentences

def get_estimator(estimator, parameters):
    """Returns a sklearn estimator object to be fitted"""
    random_state = 1
    if estimator == "SVC":
        if parameters is not None:
            return SVC(random_state=random_state, **parameters)
        else:
            return SVC(random_state=random_state)
    elif estimator == "LR":
        if parameters is not None:
            return LogisticRegression(random_state=random_state, **parameters)
        else:
            return LogisticRegression(random_state=random_state)
    elif estimator == "RFC":
        if parameters is not None:
            return RandomForestClassifier(random_state=random_state, **parameters)
        else:
            return RandomForestClassifier(random_state=random_state)
    elif estimator == "XGB":
        if parameters is not None:
            return XGBClassifier(random_state=random_state, **parameters)
        else:
            return XGBClassifier(random_state=random_state)
    else:
        print("unknown sklearn estimator")



class Model:
    model: Union[XGBClassifier, LogisticRegression, RandomForestClassifier, SVC] = None
    model_name: str = None
    model_path: str = None
    
    def __init__(self, model_path:str = None) -> None:
        self.model_path = model_path
        if model_path is not None:
            print(f'loading model from - {self.model_path}')
            _path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'data', 'models', self.model_path)
            self.model = pickle.load(open(_path, 'rb'))
    
    def predict(self, df: pd.DataFrame):
        X = df.drop(['token'], axis=1)
        y_hat = self.model.predict(X)
        
        df_predict = pd.DataFrame({'target':y_hat, 'token':df['token']})
        return binary_target_to_sentences(df_predict)

