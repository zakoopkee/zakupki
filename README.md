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

**Цель**: Чтобы владелец закупки еще до ее окончания мог определить по имеющимся данным, будет ли его закупка успешной.

Решение -- [zakupki_classification_with_one_sup.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_classification_with_one_sup.ipynb) (на полных данных), [zakupki_classification_without_one_sup.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_classification_without_one_sup.ipynb) (на данных без процедуры "Закупка у единственного поставщика")

Проведено сравнение алгоритмов `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`.

Используемые метрики: `precision`, `recall`, `f1-score`.

### Предсказание числа кандидатов закупки

**Задача**: По данным закупки надо предсказать число кандидатов на закупку.

**Цель**: Чтобы владелец еще до окончания торгов знал о возможной (не)популярности закупки и мог обеспечить ее успешность.

Решение -- [zakupki_classification.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_classification.ipynb), [zakupki_regression_r2_score.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_regression_r2_score.ipynb)

Проведено сравнение алгоритмов `DummyRegressor`, `DecisionTreeRegressor`, `SVR`, `KNeighborsRegressor`,  `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`.

Используемые метрики: `MAE`, `MSE`, `r2`.

### Предсказание числа кандидатов закупки с использованием нейронных сетей

**Задача**: По данным закупки надо предсказать число кандидатов на закупку.

**Цель**: Чтобы владелец еще до окончания торгов знал о возможной (не)популярности закупки и мог обеспечить ее успешность.

Решение -- [zakupki_regression_nn.ipynb](https://github.com/zakoopkee/zakupki/blob/master/zakupki_regression_nn.ipynb)

В качестве исходной структуры нейронной сети была взята сеть с тремя входами (title, org_name, categorical features), затем было проведено сравнение различных подходов к предобработке данных, а также различных возможных слоев нейронной сети:
* Предобработка текстовых данных: нормализация, BPE, char based RNN
* Манипуляции с текстовыми энкодерами: ConvEncoder, GruEncoder, LstmEncoder, bidirectional layer, several rnn layers, batchnorm in encoder, pretrained embedding
* Пулинг в текстовых энкодерах: MaxPooling, AvgPooling, SoftmaxPooling, AttentivePooling
* Манипуляции с категориальным энкодером: отсутствие/присутствие цены во входных данных, отсутствие/присутствие самого энкодера в структуре сети
* loss: `MSE`, `r2`
