Данный репозиторий создан в рамках хакатона "Цифровой прорыв". 

## Поставленная задача:
Внедрить чат-бота с искусственным интеллектом, что поспособствует повышению скорости и качества при работе сотрудников МФЦ с клиентами.

## Описание файлов:
Данный репозиторий содержит файлы, необходимые для запуска чат бота. В файле `app.py` содержится скрипт веб-приложения, позволяющего итерактивно взаимодействовать с ботом. `Dockerfile` содержит описание конфигурации для докера. Папка `storage.zip` нужна для сохранения фидбека пользователя. `requirements.txt` содержит все необходимые зависимости для воспроизведения кода. В архиве `data.zip` содержаться данные, составляющие внутреннюю базу знаний.

## Процесс запуска:
1. Скачать веса модели по ссылке [веса на Яндекс диске](https://disk.yandex.ru/d/TYXsBveHRjMKDA) и положить их в одну папку с проектом
2. Распаковать директории `storage.zip` и `data.zip`.
4. Запустить докер команду
    ```json
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```



Лингвистическая модель, нужная для инференса лежит на 
