import re
import numpy as np
import pandas as pd
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

class StatusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_status='for sale'):
        self.patterns = {
            re.compile(r' / auction|.*Auction.*', re.IGNORECASE): 'Auction',
            re.compile(r'C|Option Contract.*', re.IGNORECASE): 'Contract',
            re.compile(r'.*(Conting|Ct|CT|Coming soon|Back on Market).*'): 'Contingent',
            re.compile(r"Pf|Pre-foreclosure.*", re.IGNORECASE): 'Pre Foreclosure',
            re.compile(r"(?i)Foreclos*"): 'Foreclosure',
            re.compile(r'New.*', re.IGNORECASE): 'New',
            re.compile(r'(?i).*pending.*|P|Active Option.'): 'Pending',
            re.compile(r'(?i)For sale.*'): 'for sale',
            re.compile(r'(?i).*Sold.*'): 'Sold',
            re.compile(r'Lease/*|Apartment for rent.*', re.IGNORECASE): 'for rent',
            re.compile(r'(?i).*Backup.*|Accepted|Uc|U\sUnder\sContract|.*Under.*'): 'Under Contract',
            re.compile(r'(?i).*activ.*|Re activated.*'): 'Active'
        }
        self.mode_status = mode_status

    def fit(self, X, y=None):
        # Здесь можно вычислить и сохранить любые значения, необходимые для трансформации
        # Но в данном случае mode_status уже установлен в __init__
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Применение паттернов
        X_transformed['status2'] = X_transformed['status'].astype(str).apply(
            lambda x: next((replacement for pattern, replacement in self.patterns.items() if pattern.match(x)), x)
        )

        # Замена пустых значений на mode_status
        X_transformed['status2'].fillna(self.mode_status, inplace=True)
        X_transformed['status2'] = X_transformed['status2'].replace('nan', self.mode_status)

        # Удаляем исходный столбец 'status'
        X_transformed.drop(['status'], axis=1, inplace=True)

        return X_transformed

class StreetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_street='Address Not Disclosed'):
        self.mode_street = mode_street

    def fit(self, X, y=None):
        # В этом трансформере не требуется дополнительных вычислений при обучении,
        # так как mode_street уже установлен в __init__
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Замена пропущенных значений на mode_street
        X_transformed['street'] = X_transformed['street'].fillna(self.mode_street)

        return X_transformed


class PropertyTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.patterns = {
            r"(?i)\b(multi|condo|Condominium|Townhome|row home|co-op|coop|apartment|High Rise|Flat)\b": "multi-family",
            r"(?i)\b(Duplex|Triplex|Townhouse|townhouse|Cooperative|Penthouse|Bungalow|Multiple)\b": "multi-family",
            r"(?i)\b(Manufactured|Mfd|Mobile)\b": "Manufactured housing",
            r"(?i)\b(Single|story|Stories|transitional|Land|Ranch|Ranches|Contemporary|Colonial|Residential|Traditional|Garden Home|Cape Cod|Spanish|Mediterranean|Cluster Home|Florida|Tudor|SingleFamilyResidence|Craftsman|Cottage)\b": "one-to-four"
        }
        self.keywords = ["multi-family", "Manufactured housing", "one-to-four"]

    def transform(self, X, y=None):
        # Применение паттернов
        X['propertyType'] = X['propertyType'].replace(self.patterns, regex=True)

        # Создание нового признака
        X['propertyType_New'] = X['propertyType'].apply(
            lambda x: next((keyword for keyword in self.keywords
                            if re.search(keyword, str(x), re.IGNORECASE)), 'one-to-four')
        ).fillna('')

        # Удаление исходного столбца 'propertyType'
        X = X.drop(['propertyType'], axis=1)
        return X

    def fit(self, X, y=None):
        return self

class BathsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Извлечение только целых чисел из признака 'baths'
        X['baths'] = X['baths'].str.extract(r'(\d+)').astype(float)

        # Замена всех пустых значений на 0
        X['baths'].fillna(0, inplace=True)

        # Преобразование типа данных на целочисленный
        X['baths'] = X['baths'].astype(int)
        return X

class CityStateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Словарь для сопоставления штатов и модальных городов
        self.state_to_city = {
            'AL': 'Bryant', 'AZ': 'Glendale', 'BA': 'Unknown', 'CA': 'Los Angeles',
            'CO': 'Denver', 'DC': 'Washington', 'DE': 'Playa', 'FL': 'Miami',
            'Fl': 'Tamarac', 'GA': 'Atlanta', 'IA': 'Clear Lake', 'IL': 'Chicago',
            'IN': 'Indianapolis', 'KY': 'Hazard', 'MA': 'Boston', 'MD': 'Bethesda',
            'ME': 'Fairfield', 'MI': 'Detroit', 'MO': 'Saint Louis', 'MS': 'Cleveland',
            'MT': 'Fairfield', 'NC': 'Charlotte', 'NJ': 'Lakewood', 'NV': 'Las Vegas',
            'NY': 'Brooklyn', 'OH': 'Cleveland', 'OK': 'Cleveland', 'OR': 'Portland',
            'OS': 'Foreign Country', 'OT': 'Other', 'PA': 'Philadelphia',
            'SC': 'Fort Mill', 'TN': 'Nashville', 'TX': 'Houston', 'UT': 'Salt Lake City',
            'VA': 'Arlington', 'VT': 'Fairfax', 'WA': 'Seattle', 'WI': 'Milwaukee'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Заполняем пропущенные значения ������ородов на основе штата
        X_transformed['city'] = X_transformed.apply(
            lambda row: self.state_to_city[row['state']] if pd.isna(row['city']) else row['city'], axis=1
        )

        return X_transformed

class HomeFactsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Десериализация содержимого признака 'homeFacts'
        X['homeFacts'] = X['homeFacts'].apply(ast.literal_eval)

        # Создание новых признаков на основе десериализованных значений
        X['YearBuilt'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][0]['factValue'])
        X['RemodeledYear'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][1]['factValue'])
        X['Heating'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][2]['factValue'])
        X['Cooling'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][3]['factValue'])
        X['Parking'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][4]['factValue'])
        X['LotSize'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][5]['factValue'])
        X['PricePerSqft'] = X['homeFacts'].apply(lambda x: x['atAGlanceFacts'][6]['factValue'])

        # Удаление старого столбца 'homeFacts'
        return X.drop(['homeFacts'], axis=1)

class CustomFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.values_to_replace_year_built = ['', '559990649990', 'No', '1', 'No Data']
        self.values_to_replace_heating_cooling_parking = ['', 'No Data', 'None']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Обработка YearBuilt
        X['YearBuilt'] = np.where(X['YearBuilt'].isin(self.values_to_replace_year_built), None, X['YearBuilt'])

        # Обработка RemodeledYear
        X['RemodeledYear'] = np.where(X['RemodeledYear'] == 0, np.nan, X['RemodeledYear']).astype('object')

        # Обработка Heating, Cooling, Parking
        X['Heating'] = X['Heating'].replace(self.values_to_replace_heating_cooling_parking, 0).replace('[^0]', 1, regex=True).fillna(0).astype(int)
        X['Cooling'] = X['Cooling'].replace(self.values_to_replace_heating_cooling_parking, 0).replace('[^0]', 1, regex=True).fillna(0).astype(int)
        X['Parking'] = X['Parking'].replace(self.values_to_replace_heating_cooling_parking, 0).replace('[^0]', 1, regex=True).fillna(0).astype(int)

        # Извлечение чисел из LotSize и PricePerSqft
        X['LotSize'] = X['LotSize'].str.extract('(\d+.\d+|\d+)', expand=False)
        X['PricePerSqft'] = X['PricePerSqft'].str.extract('(\d+.\d+|\d+)', expand=False)

        return X

class SchoolsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Десериализация содержимого 'schools'
        X['schools'] = X['schools'].apply(ast.literal_eval)

        # Создание новых признаков
        X['rating'] = X['schools'].apply(lambda x: x[0]['rating'])
        X['Distance'] = X['schools'].apply(lambda x: x[0]['data']['Distance'])
        X['name_school'] = X['schools'].apply(lambda x: x[0]['name'])
        X['school_count'] = X['name_school'].apply(len)

        # Расчет суммы расстояний и средней дистанции до школ
        X['sum_distance'] = X['Distance'].apply(lambda x: sum(float(re.findall(r'\d+\.\d+', val)[0]) for val in x if re.findall(r'\d+\.\d+', val)) if x and any(re.findall(r'\d+\.\d+', val) for val in x) else None)
        X['Avr_distance'] = X['sum_distance'] / X['school_count']
        X['MinDistance'] = X['Distance'].apply(lambda x: min(float(re.findall(r'\d+\.\d+', val)[0]) for val in x if re.findall(r'\d+\.\d+', val)) if x and any(re.findall(r'\d+\.\d+', val) for val in x) else None)

        # Вычисление среднего рейтинга
        X['AverageRating'] = X['rating'].apply(self.calculate_rating)

        # Удаление вспомогательных столбцов
        return X.drop(['sum_distance', 'schools', 'rating', 'Distance', 'name_school'], axis=1)

    def calculate_rating(self, row):
        ratings = []
        for val in row:
            if re.match(r'^[\d./]+$', val):
                val = re.sub(r'/10$', '', val)
                try:
                    ratings.append(float(val))
                except ValueError:
                    pass
        return np.mean(ratings) if ratings else np.nan

class LotSizeSqftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sqft_median = 6.369  # Пример медианного значения, измените по необходимости

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Очистка и преобразование LotSize
        X['LotSize'] = X['LotSize'].str.replace(',', '.')
        X['LotSize'] = pd.to_numeric(X['LotSize'], errors='coerce')

        # Обработка sqft
        X['sqft'] = X['sqft'].apply(lambda x: re.sub(r'\D', '', str(x)))
        X['sqft'] = X['sqft'].replace('', np.nan).astype(float)
        X['sqft'].fillna(self.sqft_median, inplace=True)

        return X

class ZipcodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.zipcode_mode = 32137  # Пример значения по умолчанию

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Применение функции для извлечения числового значения из строки индекса
        X['zipcode'] = X['zipcode'].apply(self.extract_zipcode_number)

        # Замена значений '0' и '00000' на NaN и заполнение пропущенных значений модой
        X['zipcode'].replace({'0': np.nan, '00000': np.nan}, inplace=True)
        X['zipcode'].fillna(self.zipcode_mode, inplace=True)

        # Преобразование столбца в int64
        X['zipcode'] = X['zipcode'].astype('int64')

        return X

    @staticmethod
    def extract_zipcode_number(zipcode):
        match = re.match(r'^(\d+)-?', str(zipcode))
        return match.group(1) if match else np.nan

class BedsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_beds_new = 3.0  # Пример значения по умолчанию

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создание условного признака для фильтрации мусорных значений
        X['temp'] = X['beds'].astype(str).str.contains(r'acre|acres|sqft', case=False, regex=True).astype(int)

        # Создание нового признака 'beds_new'
        X['beds_new'] = X.apply(lambda row: self.extract_clean_beds(row['beds'], row['temp']), axis=1)

        # Удаление вспомогательных столбцов
        X = X.drop(['beds', 'temp'], axis=1)

        # Замена пропущенных значений на медиану
        X['beds_new'] = pd.to_numeric(X['beds_new'], errors='coerce').fillna(self.median_beds_new)

        return X

    @staticmethod
    def extract_clean_beds(beds, temp):
        if temp == 0:
            nums = str(beds).split()
            for num in nums:
                if num.isdigit():
                    return num
        return np.nan

class StoriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stories_median = 1.0  # Устанавливаем значение медианы по умолчанию

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Паттерны для преобразования строк в числа
        patterns = {
            r"(?i)\b(One)\b": "1",
            r"(?i)\b(Two)\b": "2",
            r"(?i)\b(Three)\b": "3",
            r"(?i)\b(Four)\b": "4",
            r"(?i)\b(Five)\b": "5",
            # Другие паттерны по необходимости
        }

        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Применение паттернов к столбцу 'stories'
        X_transformed['stories'] = X_transformed['stories'].replace(patterns, regex=True)

        # Извлечение только числовых значений и преобразование в числа
        X_transformed['stories'] = X_transformed['stories'].apply(lambda x: re.sub(r'\D', '', str(x)))
        X_transformed['stories'] = pd.to_numeric(X_transformed['stories'], errors='coerce')

        # Замена пустых значений на медианное значение
        X_transformed['stories'].fillna(self.stories_median, inplace=True)

        return X_transformed

class CombinedMlsIDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_mlsID='NO MLS'):
        self.mode_mlsID = mode_mlsID

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Объединение 'MlsId' и 'mls-id' в 'mlsID'
        X_transformed = X.copy()
        X_transformed['mlsID'] = X_transformed['MlsId'].fillna(X_transformed['mls-id'])

        # Удаление исходных столбцов 'MlsId' и 'mls-id'
        X_transformed = X_transformed.drop(['MlsId', 'mls-id'], axis=1)

        # Замена пустых значений в 'mlsID' на заданный режим
        X_transformed['mlsID'].fillna(self.mode_mlsID, inplace=True)

        return X_transformed

class PoolTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Функция для преобразования значений
        replace_func = lambda x: '1' if x in ['Yes', 'yes'] else '0'

        # Применение функции к столбцам
        X['PrivatePool'] = X['PrivatePool'].map(replace_func)
        X['private pool'] = X['private pool'].map(replace_func)

        # Объединение значений в один столбец
        X['Private_Pool'] = np.where(X['PrivatePool'] == '1', X['PrivatePool'], X['private pool'])

        # Преобразование типа столбца 'Private_Pool' в int
        X['Private_Pool'] = X['Private_Pool'].astype(int)

        # Удаление исходных столбцов
        return X.drop(['PrivatePool', 'private pool'], axis=1)

'''class YearBuiltTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_year=1980):
        self.median = median_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Преобразование пустых строк и других нечисловых значений в NaN
        X_transformed = X.copy()
        X_transformed['YearBuilt'] = pd.to_numeric(X_transformed['YearBuilt'], errors='coerce')

        # Заполнение NaN медианным значением
        X_transformed['YearBuilt'].fillna(self.median, inplace=True)

        # Преобразование данных в целочисленный тип
        X_transformed['YearBuilt'] = X_transformed['YearBuilt'].astype(int)

        return X_transformed'''

class YearBuiltTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_year=1980):
        self.median_year = median_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed['YearBuilt'] = pd.to_numeric(X_transformed['YearBuilt'], errors='coerce')
        X_transformed['YearBuilt'].fillna(self.median_year, inplace=True)
        X_transformed['YearBuilt'] = X_transformed['YearBuilt'].astype(int)
        return X_transformed

class RemodeledYearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_year=1985):
        self.median_year = median_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Преобразование пустых строк в np.nan
        X_transformed = X.copy()
        X_transformed['RemodeledYear'] = X_transformed['RemodeledYear'].replace('', np.nan)

        # Преобразование в числовой тип и замена нулей на медиану
        X_transformed['RemodeledYear'] = pd.to_numeric(X_transformed['RemodeledYear'], errors='coerce')
        X_transformed['RemodeledYear'] = X_transformed['RemodeledYear'].replace(0, self.median_year)

        # Заполнение пропущенных значений медианой
        X_transformed['RemodeledYear'].fillna(self.median_year, inplace=True)

        # Преобразование в целочисленный тип
        X_transformed['RemodeledYear'] = X_transformed['RemodeledYear'].astype(int)

        return X_transformed


class LotSizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_lot_size=6.369):
        self.median_lot_size = median_lot_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Преобразование в строковый тип и очистка данных
        X_transformed['LotSize'] = X_transformed['LotSize'].astype(str).str.replace(',', '').str.replace(' ', '')

        # Преобразование в числовой тип и заполнение пропущенных значений медианой
        X_transformed['LotSize'] = pd.to_numeric(X_transformed['LotSize'], errors='coerce').fillna(self.median_lot_size)

        return X_transformed

class PricePerSqftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode=1.0):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Извлечение числовых значений из строк
        X_transformed['PricePerSqft'] = X_transformed['PricePerSqft'].astype(str).apply(lambda x: re.findall(r'\d+', x))
        X_transformed['PricePerSqft'] = X_transformed['PricePerSqft'].apply(lambda x: float(x[0]) if x else np.nan)

        # Замена пустых значений на моду
        X_transformed['PricePerSqft'].fillna(self.mode, inplace=True)
        
        return X_transformed

class AvrDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_avr_distance=1.7666):
        self.median_avr_distance = median_avr_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Преобразование в числовой тип и заполнение пропущенных значений медианой
        X_transformed['Avr_distance'] = pd.to_numeric(X_transformed['Avr_distance'], errors='coerce').fillna(self.median_avr_distance)
        
        return X_transformed

class MinDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_min_distance=0.7):
        self.median_min_distance = median_min_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для б����зоп��сного преобразования
        X_transformed = X.copy()

        # Преобразование в числовой тип и заполнение пропущенных значений медианой
        X_transformed['MinDistance'] = pd.to_numeric(X_transformed['MinDistance'], errors='coerce').fillna(self.median_min_distance)

        return X_transformed

class AverageRatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, median_average_rating=5.0):
        self.median_average_rating = median_average_rating

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Создаем копию DataFrame для безопасного преобразования
        X_transformed = X.copy()

        # Преобразование значения среднего рейтинга, заполнение пустых значений медианным значением
        X_transformed['AverageRating'] = pd.to_numeric(X_transformed['AverageRating'], errors='coerce')
        X_transformed['AverageRating'].fillna(self.median_average_rating, inplace=True)

        return X_transformed

class MlsIDFireplaceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Преобразование 'mlsID'
        X['mlsID'] = X['mlsID'].apply(lambda x: 1 if pd.notnull(x) else 0)

        # Преобразование 'fireplace'
        replace_func = lambda x: '1' if x in ['', 'No', 'No Fireplace', 'None', 'Not Applicable'] else '0'
        X['fireplace'] = X['fireplace'].map(replace_func).astype(int)

        return X

'''class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in ['city', 'street', 'state', 'status2', 'propertyType_New']:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X, y=None):
        for col, encoder in self.encoders.items():
            X[col] = encoder.transform(X[col])
        return X'''

'''class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None):
        self.features_to_encode = features_to_encode
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.features_to_encode:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            X_transformed[col] = encoder.transform(X[col])
        return X_transformed'''

'''class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None):
        self.features_to_encode = features_to_encode
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.features_to_encode:
            if col in X:
                self.encoders[col] = LabelEncoder().fit(X[col].fillna('Missing'))
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            if col in X_transformed:
                X_transformed[col] = encoder.transform(X_transformed[col].fillna('Missing'))
        return X_transformed'''

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None):
        self.features_to_encode = features_to_encode
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.features_to_encode:
            # Если данные не категориальные, преобразуем их в строки
            if not pd.api.types.is_categorical_dtype(X[col]):
                X[col] = X[col].astype(str)
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            # Преобразование данных в строки, если они не категориальные
            if not pd.api.types.is_categorical_dtype(X[col]):
                X_transformed[col] = X[col].astype(str)
            X_transformed[col] = encoder.transform(X_transformed[col])
        return X_transformed

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.features_to_drop, axis=1, errors='ignore')

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Обучаем StandardScaler на данных X
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        # Применяем масштабирование к данным X
        X_scaled = self.scaler.transform(X)
        return X_scaled

