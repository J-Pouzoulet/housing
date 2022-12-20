from sklearn.base import BaseEstimator, TransformerMixin
import patsy
import pandas as pd
import numpy as np

class FormulaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, formula):
        self.formula = formula
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=columns)
    

class NoTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X
    
class Float16Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[['housing_median_age','pop_per_rooms', 'population', 'median_income']] = X[['housing_median_age','pop_per_rooms', 'population','median_income']].astype(np.float16)
        return X
    
class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[['median_house_value']] = X[['median_house_value']].astype(np.float32)
        return X
    

class DFTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)

class ColNameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, col_names = ['housing_median_age','pop_per_rooms', 'population','median_income', 'zip_simplified']):
        X.columns = col_names
        return X
    
class ColNameTransformer2(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, col_names = ['median_house_value','housing_median_age','pop_per_rooms', 'population', 'median_income', 'zip_simplified']):
        X.columns = col_names
        return X