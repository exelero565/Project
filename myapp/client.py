# для запуска в докере
# python test_request.py

import requests

# URL вашего сервера
url = 'http://localhost:5001/upload'

# Открытие файла для чтения в бинарном режиме
with open('test_datt.csv', 'rb') as file:
    files = {'file': file}
    # Отправка POST-запроса с файлом
    response = requests.post(url, files=files)

# Вывод ответа от сервера
print(response.text)