from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle
import numpy as np

import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import median_absolute_error
import optuna
from sklearn import ensemble
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import xgboost as xgb
import time
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from transformers import (
    StatusTransformer,
    StreetTransformer,
    PropertyTypeTransformer,
    BathsTransformer,
    CityStateTransformer,
    HomeFactsTransformer,
    CustomFeaturesTransformer,
    SchoolsTransformer,
    LotSizeSqftTransformer,
    ZipcodeTransformer,
    BedsTransformer,
    StoriesTransformer,
    CombinedMlsIDTransformer,
    PoolTransformer,
    YearBuiltTransformer,
    RemodeledYearTransformer,
    LotSizeTransformer,
    PricePerSqftTransformer,
    AvrDistanceTransformer,
    MinDistanceTransformer,
    AverageRatingTransformer,
    CategoricalEncoder,
    MlsIDFireplaceTransformer,
    FeatureDropper
)

app = Flask(__name__)

# Загрузка pipeline и модели
pipeline = joblib.load('pipeline.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return "Привет! Это мой веб-сервис для прогнозирования стоимости недвижимости."

@app.route('/upload', methods=['POST'])
def upload_file():
    # Получение файла из запроса
    uploaded_file = request.files['file']
    if uploaded_file:
        # Чтение данных из CSV-файла в DataFrame
        df = pd.read_csv(uploaded_file)

        # Обработка данных с помощью pipeline и модели
        pipeline.fit(df) 
        processed_data = pipeline.transform(df)
        predictions = model.predict(processed_data)
        predictions = np.exp(predictions)
        # Отправка результата
        return jsonify(predictions.tolist())
    else:
        return jsonify({'error': 'Нет файла в запросе'}), 400

if __name__ == '__main__':
    app.run('localhost', 5001)