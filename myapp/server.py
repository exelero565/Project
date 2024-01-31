from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import dill

# Импорт классов трансформеров
from transformers2 import (
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
    FeatureDropper,
    StandardScalerTransformer,
    CategoricalEncoder
)

app = Flask(__name__)

# Создание экземпляров трансформеров
transformers = [
    StatusTransformer(),
    StreetTransformer(),
    PropertyTypeTransformer(),
    BathsTransformer(),
    CityStateTransformer(),
    HomeFactsTransformer(),
    CustomFeaturesTransformer(),
    SchoolsTransformer(),
    LotSizeSqftTransformer(),
    ZipcodeTransformer(),
    BedsTransformer(),
    StoriesTransformer(),
    CombinedMlsIDTransformer(),
    PoolTransformer(),
    YearBuiltTransformer(),
    RemodeledYearTransformer(),
    LotSizeTransformer(),
    PricePerSqftTransformer(),
    AvrDistanceTransformer(),
    MinDistanceTransformer(),
    AverageRatingTransformer(),
    MlsIDFireplaceTransformer(),
    FeatureDropper(features_to_drop=['Avr_distance', 'target', 'Cooling', 'Parking'])
]

# Экземпляр CategoricalEncoder
categorical_encoder = CategoricalEncoder(features_to_encode=['city', 'street', 'state', 'status2', 'propertyType_New'])

# Загрузка экземпляра scaler
with open('scaler2.pkl', 'rb') as f:
    scaler = dill.load(f)

@app.route('/')
def index():
    return "Привет! Это мой веб-сервис для обработки данных о недвижимости."

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Применение трансформеров к данным через цикл
        for transformer in transformers:
            df = transformer.transform(df)
        # Применение категоризации
        categorical_encoder.fit(df)
        df = categorical_encoder.transform(df)
        # Масштабирование данных
        df = scaler.transform(df)
        # Загрузка модели и получение предсказаний
        model = joblib.load('model.pkl')
        predictions = model.predict(df)
        # Разлогарифмирование предсказаний
        predictions = np.exp(predictions)
        return jsonify({'predictions': predictions.tolist()})
    else:
        return jsonify({'error': 'Нет файла в запросе'}), 400
          
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
