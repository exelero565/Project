# DS_project_property


### Описание проекта
В данное проекте подготавливаем датасет для построения ML модели, предсказывающую стоимость недвижимости.

### Краткая информация о данных
Данные представляют собой таблицу в формате CSV, в каждой строке которой содержится информация об уникальном объекте недвижимости.


**Основные цели проекта:**
1. Провести разведывательный анализ и очистку исходных данных.
2. Выделить наиболее значимые факторы, влияющие на стоимость недвижимости.
3. Построить модель для прогнозирования стоимости недвижимости.
4. Разработать небольшой веб-сервис, на вход которому поступают данные
о некоторой выставленной на продажу недвижимости, а сервис прогнозирует его стоимость.

### Метрика качества 
Для регрессии - Средняя квадратичная ошибка (MSE), Коэффициент детерминации (R-squared).

### Выводы:  
1. - удалось провести разведывательный анализ и очистить данные от выбрасов и жаргонных сокращений.
2. - удалось выделить наиболее значимые факторы, влияющие на стоимость недвижимости:
'PricePerSqft' - означает цену недвижимости за квадратный фут, выделяется как наиболее значимый;
'sqft' и 'zipcode' также показывают значительную важность, но они уступают 'PricePerSqft'. Это подчёркивает, что площадь объекта и местоположение при оценке стоимости, но именно соотношение цены и площади играет ключевую роль.
3. - удалось выбратьи и построить оптимальную модель и подобрать гиперпараметры.
Хотя подход optuna дает лучшую производительность модели, он значительно более времязатратен. Подход RandomizedSearchCV предлагает более быстрый процесс оптимизации с небольшим уступанием в точности. Выбор между этими двумя подходами зависит от приоритетов: если важнее точность, optuna подход предпочтительнее; если важнее скорость, то RandomizedSearchCV подход может быть более подходящим. Важно отметить, что различия в производительности не так велики, что может означать, что более быстрый подход является разумным компромиссом.
4. - Удалось разработать небольшой веб-сервис, предсказывающий стоимость недвижимости.

В целом, получилось преобразовать, подготовить данные, подобрать оптимальные гиперпараметры и создать вэб-сервис по предсказанию стоиомсти недвижимости.

## Установка вэб сервиса
docker pull exelero565/my_app_property

## Для установки проекта следуйте шагам:

```bash
# Клонируйте репозиторий
git clone https://github.com/exelero565/Project.git

# Перейдите в директорию проекта
cd Project
