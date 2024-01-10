from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

    

class NoTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X
    
