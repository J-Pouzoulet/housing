#see folder "SKL_regression_with_pipeline" and file "workflow_with_pipeline.ipynb" for the pipeline used in the pipeline.pkl
#'''uvicorn housing_API:app --reload''' in terminal to lunch the API >>> copy and past the url in goolgle chrome


from fastapi import FastAPI
import pickle
from typing import Union

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PowerTransformer

#library to create the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
import patsy
import patsylearn
from my_transformers import NoTransformer
from my_transformers import Float16Transformer, Float32Transformer
from my_transformers import DFTransformer
from my_transformers import ColNameTransformer, ColNameTransformer2

app = FastAPI()

@app.get('/')
def prediction(housing_median_age : Union[float, None] = 40,
               pop_per_rooms : Union[float, None] = 1,
               population : Union[float, None] = 5000,
               median_income : Union[float, None] = '40000',
               zip_simplified : Union[str, None] = '925',
                ):
    
    housing_prepro = pickle.load(open('housing_prepro.pkl', 'rb'))
    housing_model = pickle.load(open('housing_model.pkl', 'rb'))

    X = pd.read_csv('housing_API.csv')
    X = X.drop(columns=['median_house_value'])
    X['zip_simplified'] = X['zip_simplified'].astype('str')
    X = housing_prepro.transform(X)

    formula = 'housing_median_age * np.log(pop_per_rooms) * np.log(population) * median_income * zip_simplified'
    patsify = patsylearn.PatsyTransformer(formula, add_intercept=True)
    patsify.fit(X)
    
    df_X = pd.DataFrame({'housing_median_age' : [housing_median_age], 'pop_per_rooms' : [pop_per_rooms], 'population' : [population],'median_income' : [median_income], 'zip_simplified' : [zip_simplified]})
    df_X = housing_prepro.transform(df_X)
    df_X = patsify.transform(df_X)
    prediction = housing_model.predict(df_X)
    return {'median_house_value' : f'{round(prediction[0])} USD$'}

#query example >>>   ?housing_median_age=20&pop_per_rooms=2&population=2000&median_income=50000&zip_simplified=920
