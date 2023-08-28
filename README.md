Данный репозиторий создан в рамках хакатона "Цифровой прорыв". 

## Поставленная задача:
Внедрить чат-бота с искусственным интеллектом, тем самым поспособствав повышению скорости и качества при работе сотрудников МФЦ с клиентами.

## Описание файлов:
Данный репозиторий содержит файлы, необходимые для запуска чат бота. В файле `app.py` содержится скрипт веб-приложения, позволяющего итерактивно взаимодействовать с ботом. `Dockerfile` содержит описание конфигурации для докера. Папка `storage.zip` нужна для сохранения фидбека пользователя. `requirements.txt` содержит все необходимые зависимости для воспроизведения кода. В архиве `data.zip` содержаться данные, составляющие внутреннюю базу знаний. Jupyter-ноутбуки `parse-MFC.ipynb` и `mapping.ipynb` содержат процедуры парсинга HTML-страниц МФЦ и создания маппинга номера услуги и ее полного наименования соответственно.

## Процесс запуска:
#### Первый способ (через GitHub)
1. Скачать проект с репозитория на GitHub;
2. Скачать веса модели по ссылке [веса на Яндекс диске](https://disk.yandex.ru/d/TYXsBveHRjMKDA) и положить их в одну папку с проектом;
3. Распаковать директории `storage.zip` и `data.zip`;
4. Запуск файла app.py, приведет к запуску чат бота.


#### Второй способ (через GitHub + docker)
1. Исполнение 3-ех первых шагов из "Первого способа";
2. Прописать в терминале следующие docker команды:
   ```bash
   # запуск сборки образа с вашим название image_name
   docker build -t image_name .
   # запуск контейнера с вашим названием name из образа image_name
   docker run -it --name name -p 127.0.0.1:7860:7860 --rm image_name
   ```
3. После исполения появится ссылка при переходе по которой будет запущен чат бот.

#### Третий способ: (через DockerHub)
1. Скачать проект с репозитория на DockerHub, применяя в docker команду:
   ```bash
    docker pull paranoid21/team_pressure:latest
   ```
