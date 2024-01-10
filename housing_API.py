#see housing_main_notebook.ipynb for the pipeline used in the housing_trained.pkl
#'''uvicorn housing_API:app --reload''' in terminal to lunch the API >>> copy and past the url in google chrome


from fastapi import FastAPI
import pickle
from typing import Union
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from my_transformers import NoTransformer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()

@app.get('/')
def prediction(longitude : Union[float, None] = -118,
                latitude : Union[float, None] = 34,
                housing_median_age : Union[float, None] = 40,
                total_rooms : Union[float, None] = 5,
                total_bedrooms : Union[float, None] = 2,
                population : Union[float, None] = 5000,
                households : Union[float, None] = 3,
                median_income : Union[float, None] = 40000,
                pop_per_rooms: Union[float, None] = 1):
    
    model = pickle.load(open('housing_trained.pkl', 'rb'))
    
    df_X = pd.DataFrame(data = {'longitude' : [longitude],
                                'latitude' : [latitude],
                                'housing_median_age' : [housing_median_age],
                                'total_rooms' : [total_rooms],
                                'total_bedrooms' : [total_bedrooms],
                                'population' : [population],
                                'households' : [households],
                                'median_income' : [median_income],
                                'pop_per_rooms': [pop_per_rooms]})
                        
    prediction = model.predict(df_X)
    return {'median_house_value' : f'{round(prediction[0])} USD$'}

#query example to add in the url>>>   ?longitude=-126&latitude=33&housing_median_age=20&total_rooms=3&total_bedrooms=1&population=2000&median_income=50000&pop_per_rooms=2
