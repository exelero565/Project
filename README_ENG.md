

### Project Description

In this project, we prepare a dataset for building an ML model that predicts real estate prices.

### Brief Data Information
The data is presented in a CSV table format, where each row contains information about a unique real estate object.

**Main objectives of the project:**
1. Conduct exploratory data analysis and clean the initial data.
2. Identify the most significant factors affecting real estate prices.
3. Build a model to forecast real estate prices.
4. Develop a small web service that receives data about a piece of real estate listed for sale and predicts its price.

### Quality Metric
For regression - Mean Squared Error (MSE), Coefficient of Determination (R-squared).

### Conclusions:
1. - Successfully conducted exploratory analysis and cleaned the data from outliers and jargon abbreviations.
2. - Managed to identify the most significant factors affecting real estate prices:
'PricePerSqft' - signifies the real estate price per square foot, highlighted as the most significant;
'sqft' and 'zipcode' also show considerable importance but are secondary to 'PricePerSqft'. This emphasizes that the area of the object and the location are important in evaluating the cost, but the ratio of price to area plays a key role.
3. - Succeeded in selecting and building the optimal model and adjusting the hyperparameters.
Although the optuna approach provides better model performance, it is significantly more time-consuming. The RandomizedSearchCV approach offers a quicker optimization process with a slight compromise in accuracy. The choice between these two approaches depends on priorities: if accuracy is more critical, the optuna approach is preferable; if speed is more critical, then the RandomizedSearchCV approach might be more suitable. It's important to note that the performance differences are not so significant, which may mean that the quicker approach is a reasonable compromise.
4. - Managed to develop a small web service predicting real estate prices.

Overall, it was possible to transform, prepare the data, select optimal hyperparameters, and create a web service for predicting real estate prices.

## Web Service Installation
```bash
docker pull exelero565/my_app_property
```

## Follow these steps to install the project:

```bash
# Клонируйте репозиторий
git clone https://github.com/exelero565/Project.git

# Перейдите в директорию проекта
cd Project

# Создание и активация виртуального окружения (рекомендуется для изоляции зависимостей проекта):
python -m venv venv
# Для Windows
venv\Scripts\activate
# Для Unix или MacOS
source venv/bin/activate

# Установка необходимых зависимостей:
pip install -r requirements.txt
```

### License
The project is distributed under the MIT license. You can freely use and distribute this code for personal and commercial purposes with a mandatory link to the author.

### Acknowledgments
Credits to data providers, contributors, and any references used in the development of this project.

## Author
https://github.com/exelero565
