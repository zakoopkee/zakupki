## Закупки

Решение задач "Предсказание результата закупки" и "Предсказание числа кандидатов закупки" по государственным закупкам.

### Используемые данные

Данные взяты с сайта [zakupki.gov.ru](http://www.zakupki.gov.ru/epz/main/public/home.html).

Предобработка в файле `data_preprocessing.py`, выделение фич в `feature_extractor.py`.

Взяты фичи:
* цена
* валюта
* ОГРН
* название закупки
* название организации-владельца

### Предсказание результата закупки

**Задача**: По данным закупки надо предсказать ее статус -- успех/неуспех/отмена.

**Зачем**: Чтобы владелец закупки еще до ее окончания мог определить по имеющимся данным, будет ли его закупка успешной.

Решение -- [zakupki_classification.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_classification.ipynb)

Проведено сравнение алгоритмов `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`.

Используемые метрики: `precision`, `recall`, `f1-score`.

### Предсказание числа кандидатов закупки

**Задача**: По данным закупки надо предсказать число кандидатов на закупку.

**Зачем**: Чтобы владелец еще до окончания торгов знал о возможной (не)популярности закупки и мог обеспечить ее успешность.

Решение -- [zakupki_classification.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_classification.ipynb), [zakupki_regression_r2_score.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_regression_r2_score.ipynb)

Проведено сравнение алгоритмов `DummyRegressor`, `DecisionTreeRegressor`, `SVR`, `KNeighborsRegressor`,  `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`.

Используемые метрики: `MAE`, `MSE`, `r2`.
