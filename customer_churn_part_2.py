import streamlit as st
from PIL import Image
# import joblib
# from joblib import dump, load
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import random
# import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
scaler = StandardScaler()
gbc = GradientBoostingClassifier(random_state=42)
from sklearn.metrics import roc_auc_score,f1_score, roc_curve #accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
# from sklearn.inspection import permutation_importance


# ---------------------Header---------------------
st.markdown('''<h1 style='text-align: center; color: #F64A46;'
            >Прогнозирование оттока клиентов банка</h1>''', 
            unsafe_allow_html=True)
st.markdown('''<h2 style='text-align: center; color: #F64A46;'
            >Часть 2: ML-методы</h2>''', 
            unsafe_allow_html=True)
st.markdown('''<h3 style='text-align: center; color: #30BA8F;'
            >Bank customer churn prediction: Part 2</h3>''', 
            unsafe_allow_html=True)

img_churn = Image.open('customers_3.jpeg') #
st.image(img_churn, use_column_width='auto') #width=450

st.write("""
Лабораторная работа *"Прогнозирование оттока клиентов банка"* помогает на реальном примере понять, 
как при помощи методов классического Machine Learning можно предсказывать отток клиентов из банка.
\nЛаборатрная работа состоит из *двух частей*: 
\n* **Первая часть**: анализ данных с помощью статистических методов и инструментов визуализации 
\n* **Вторая часть**: прогноз оттока клиентов с использованием методов машинного обучения""")

st.markdown('''<h2 style='text-align: left; color: black;'
            >Задача:</h2>''', 
            unsafe_allow_html=True)
st.write(""" \nПредположим, что вы - управляющий одного из отделения банка "Российкий". 
\nПоследние несколько месяцев аналитика по отделению ухудшается: клиенты стали уходить. 
Пока ушедших немного, но это уже заметно. Банковские маркетологи посчитали, что сохранить текущих клиентов дешевле, чем привлекать новых.
\n**Таким образом, нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет, с учётом исторических данные о поведении клиентов и расторжении договоров с банком.**
\n Задания *второй части лабораторной работы* будут включать 2 блока: прогноз оттока клиентов с помощью модели машинного обучения БЕЗ корректировки данных и С корректировкой дисбаланса классов клиентов.
\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")
#-------------------------Pipeline description-------------------------
img_pipeline = Image.open('churn_pipeline.png') 
st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн лабораторной работы') #width=450


pipeline_bar = st.expander("Описание пайплайна лабораторной работы:")
pipeline_bar.markdown(
    """
    \n**Этапы:**
    \n(зелёным обозначены этапы, корректировка которых доступна студенту, красным - этапы, что предобработаны и скорректированы сотрудником лаборатории)
    \n1. Сбор данных:
    \nБыл использован учебный датасет по прогнозированию оттока банковских клиентов [(ссылка на данные)](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling);
    \n2. Предобработка данных:
    \nУдаление ненужных колонок, one hot encoding категориальных переменных, заполнение пропущенных значений. С использованием библиотек [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [numpy](https://numpy.org/doc/stable/reference/index.html);
    \n3. Анализ статистических показателей и графический анализ данных:
    \nИнструменты для этого - с использованием библиотек [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html);
    \n4. Выбор baseline-модели, подбор лучших гиперпараметов, обучение и валидация модели с ними:
    \nС использованием библиотек [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html);
    \n5. Работа с дисбалансом классов: даунсемплинг данных, обучение и валидация модели:
    \nС использованием библиотек [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html);
    \n6. Сравнение результатов: если результат неудовлетворительный, проводится корректировка гиперпараметров и функций из п.4 и п.5;
    \n7. Оформление микросервиса Streamlit, выгрузка на сервер: проводится сотрудником лаборатории, используется студентами РАНХиГС
    \nС использованием библиотеки [streamlit](https://docs.streamlit.io/library/get-started).
    """)
info_bar = st.expander("Информация о применении методов машинного обучения для бизнес-задач:")
info_bar.markdown(
    """
    \nВ настоящее время крупные компании вкладывают большие средства в машинное обучение, потому что данная технология не только окупается, но и помогает найти новые подходы к реализации рутинных задач. 
    Действительно, ИИ занимает все более значимое место на рынке, но это не значит, что машины нас заменят.
    \nCреди "больших данных", которыми обладают компании и к которым применяются методы машинного обучения, есть статистика по поведению клиентов, их реакция на прошлую коммуникацию, история получения и возвратов кредитов, анкеты клиентов, параметры сотрудников, история эффективности работы персонала и другое.
    \nКоличество примеров проектов, реализуемых на базе машинного обучения, множество, и успешные кейсы будут появляться все чаще. Но главное усвоить базовые знания о том, что в действительности используют специалисты по машинному обучению, 
    и заранее просчитать, будет ли от вашего будущего ML-проекта бизнес-эффект. 
    \nЗадачи, которые решает ML в ритейле, включают в себя предсказание оттока клиентов, анализ продуктовых корзин, прогнозирование товаров в следующем чеке, распознавание ценников и товаров, прогноз закупок 
    и спроса, оптимизация закупок и логистики, планирование промо, цен и ассортимента — или это лишь малая часть.
    \nРитейл не испытывает недостатка как в наличия разных данных, так и в их глубине истории. У ритейлеров есть история продаж, статистика поведения клиентов, история промоакций, исторический ассортимент, 
    параметры товаров и магазинов, изображения ценников и товаров, история доставок и поступления товаров и многое другое. Оцифровка всего этого, чаще всего, не требуется.
    \nПохуже с данными в сфере промышленности — хотя и там они есть. Это и исторические данные с датчиков о производительности, поломках, работе бригад, данные по расходу и поставкам сырья, отгрузкам и доставкам. 
    Для производств каждый процент простоя – это существенные потери, поэтому именно способы его сокращения, как и сокращение запасов, становятся основными задачами для оптимизации. Поэтому в числе главных задач для ML здесь — предсказание поломок оборудования, 
    маркировка похожих поломок, выявление закономерностей поломок, выявление факторов на снижения производительности, оптимизация расхода сырья в производстве, оптимизация заказов и времени поставок сырья, прогноз скорости доставки. 

    \nЕще две отрасли, в которых распространены проекты на базе искусственного интеллекта, это банки и телекоммуникации. Это и управление клиентскими рисками (кредитный скоринг), и оптимизация регулярных рассылок клиентам. 
    Задачи, стоящие в этих проектах, разношерстны – от предсказания оттока клиентов до маркировки клиентов, от кросс-сейл кредитов и депозитов до предсказания крупных транзакций. [Источник](https://habr.com/ru/post/530012/)
    """)
# ---------------------Reading CSV---------------------
df = pd.read_csv('Churn_rus.csv')
# дропнем ненужное:
df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
# заэнкодим:
df_ohe = pd.get_dummies(df)
# заполним пропущенные значения:
filling = np.random.randint(0, 11, size = df_ohe['Tenure'].isna().sum())
df_ohe.loc[df_ohe['Tenure'].isna(), 'Tenure'] = filling
# разобъём на числовые и категориальные:
numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical = ['HasCrCard', 'IsActiveMember', 'Exited', 'Geography_Kazan', 'Geography_Moscow', 'Geography_Saint_Petersburg', 'Gender_Female', 'Gender_Male']

# st.markdown('''<h2 style="text-align: left; color: black;"> Блок 1:\nанализ и предобработка данных </h2>''', 
#             unsafe_allow_html=True)

# st.markdown(''' \nНиже представлены 4 опции для просмотра исходной таблицы с данными:
# \n **"Показать датасет"** - просмотр таблицы данных клиентов вашего банка
# \n **"Показать названия колонок"** - отображает расшифровку информации в таблице
# \n **"Посмотреть размер датасета"** - показывает общее количество клиентов и характеристик по ним
# \n **"Посмотреть статистические характеристики"** - демонстрирует 8 базовых статистических параметров по всем колонкам таблицы
# \n *Задание по Блоку 1:*
# \n Пользуясь вышеперечисленными опциями, ответьте на вопросы и занесите ответы в Бланк лабораторной работы:
# \n 1. Сколько всего клиентов пользуются услугами вашего отделения банка? Какое количество данных (количество признаков) собирает банк по каждому клиенту?
# \n 2. Какой средний возраст ваших клиентов?
# \n 3. Как варьируется баланс на счетах клиентов банка?
# \n 4. Доход ниже какой суммы имеет половина клиентов банка?''')

# if st.checkbox('Показать датасет'):
#   number = st.number_input('Сколько строк показать', min_value=1, max_value=df.shape[1])
#   st.dataframe(df.head(number))

# if st.checkbox('Показать названия колонок'):
#   st.write('''
#  \n• **RowNumber** — индекс строки в данных
#  \n• **CustomerId** — уникальный идентификатор клиента
#  \n• **Surname** — фамилия клиента
#  \n• **CreditScore** — кредитный рейтинг
#  \n• **Geography** — страна проживания
#  \n• **Gender** — пол клиента
#  \n• **Age** — возраст клиента
#  \n• **Tenure** — количество лет прибывания клиентом банка
#  \n• **Balance** — баланс на счёте
#  \n• **NumOfProducts** — количество продуктов банка, используемых клиентом
#  \n• **HasCrCard** — наличие кредитной карты
#  \n• **IsActiveMember** — активность клиента
#  \n• **EstimatedSalary** — предполагаемая зарплата
#  \n• **Exited** — факт ухода клиента - *целевой признак для прогнозирования*''')

# if st.checkbox('Посмотреть размер датасета'):
#   shape = st.radio(
#     "Показать количество...",
#      ('...строк', '...колонок'))
#   if shape == '...строк':
#     st.write('Количество строк (клиентов):', df.shape[0])
#   elif shape == '...колонок':
#     st.write('Количество колонок (признаков клиента):', df.shape[1])

# if st.checkbox('Посмотреть статистические характеристики'):
#   st.write(df.describe())
#   st.write('''
#  \n• **count** — количество заполненных ячеек по колонке
#  \n• **mean** — среднее значение по колонке
#  \n• **std** — стандартное отклоение: показывает среднюю степень разброса значений параметра относительно математического ожидания
#  \n• **min** — наименьшее значение по колонке
#  \n• **25%** — 25% перцентиль: показывает, что 25% всей выборки имеют значение параметра (столбца) **меньше** указанного
#  \n• **50%** — 50% перцентиль: показывает, что 50% всей выборки имеют значение параметра (столбца) **меньше** указанного
#  \n• **75%** — 75% перцентиль: показывает, что 75% всей выборки имеют значение параметра (столбца) **меньше** указанного
#  \n• **max** — наибольшее значение по колонке''')

# st.markdown('''<h2 style="text-align: left; color: black;">Блок 2: Графики и зависимости в данных: </h2>''', 
#             unsafe_allow_html=True)
# st.markdown(''' \nНиже представлены 3 варианта графиков для визуализации данных и нахождения зависимости в них:
# \n **"HistPlot"** - построить гистограмму по выбранному (одному) признаку клиентов
# \n **"Correlation Heatmap"** - построить матрицу корреляций признаков
# \n **"BoxPlot"** - построить диаграмму размаха признака 
# \n *Задание по Блоку 2:*
# \n Пользуясь вышеперечисленными графиками, ответьте на вопросы и занесите ответы в Бланк лабораторной работы:
# \n 1. Сколько клиентов имеют на своём балансе меньше 20000?
# \n 2. Кого среди клиентов больше: с кредитными картами или без них?
# \n 3. Какое количество клиентов используют только один банковский продукт? Большинство ли таких клиентов?
# \n 4. Кого среди клиентов больше: мужчин или женщин?
# \n 5. Определите, какая характеристика наиболее коррелирует с целевым признаком?
# \n 6. Из какого города, вероятнее всего, будет человек с высоким балансом на счёте?
# \n 7. С каким признаком меньше всего взаимосвязи у кредитного рейтинга клиентов?
# \n 8. Сколько лет пользуется услугами банка самый возрастной клиент?
# \n 9. Сколько лет самому богатому клиенту банка?
# \n 10. Чей возраст в среднем больше: клиентов, кто покинул банк или тех, кто остался пользоваться его услугами?

# \n Если задания показались сложными - воспользуйтесь подсказкой ниже:''')
# if st.button('Показать подсказку', key='hint1'):
#   st.write('Вопросы с 1 по 4 - на HistPlot, с 5 по 7 - на Correlation Heatmap, с 8 по 10 - на BoxPlot')

# #-----------------Preprocessing for students--------------------
# # дропнем ненужное:
# df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
# # заэнкодим:
# df_ohe = pd.get_dummies(df)
# # заполним пропущенные значения:
# filling = np.random.randint(0, 11, size = df_ohe['Tenure'].isna().sum())
# df_ohe.loc[df_ohe['Tenure'].isna(), 'Tenure'] = filling
# # разобъём на числовые и категориальные:
# numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
# categorical = ['HasCrCard', 'IsActiveMember', 'Exited', 'Geography_Kazan', 'Geography_Moscow', 'Geography_Saint_Petersburg', 'Gender_Female', 'Gender_Male']

# #-----------------HistPlot--------------------
# if st.checkbox('Построить гистораму распределения признака (HistPlot)'):
#   st.write('*HistPlot* - показывает распредление числовых значений объекта')
#   option = st.multiselect('Выберите один признак:', df_ohe.select_dtypes(exclude=['object']).columns)
#   if not option:
#     st.write('*Признак для посторения гистограммы не выбран*')
#   else:
#     fig, ax = plt.subplots()
#     fig = plt.figure(figsize=(20,10))
#     plt.ticklabel_format(style='plain')
#     ax = sns.histplot(data = df_ohe, x = df_ohe[option[0]], kde = True)
#     sns.set_theme(style='whitegrid') # , palette="tab10"
#     # sns.color_palette("husl", 8)
#     st.pyplot(fig)

# #------------------HeatMap--------------------
# if st.checkbox('Построить график корреляции признаков (Correlation Heatmap)'):
#   st.write('*Correlation Heatmap* - графическое представление корреляционной матрицы, которая показывает зависимость между числовыми объектами')
#   expander_bar = st.expander('Подробнее о корреляционной матрице')
#   expander_bar.info(''' Корреляция - статистическая взаимосвязь двух или более переменных. Изменения значений одной переменной сопутствуют изменениям другой.
#   \nВ нашем слуаем зависимость выражется на промежутке от -1 до 1. 
#   \nЧем ближе значение к 1, тем сильнее прямая зависимость: увеличивая значения одной переменной, увеличивается значение и второй.
#   \nЧем ближе значение к -1, тем сильнее обратная зависимость: увеличивая значение одной переменной, уменьшается значение второй и наоброт. 
#   ''')
#   fig, ax = plt.subplots(figsize=(20,10)) 
#   ax = sns.heatmap(df_ohe.corr(),vmin=-1, vmax=1, annot=True, cmap='vlag',
#                    center = 0, fmt='.1g', linewidths=1, linecolor='black')
#   plt.xticks(rotation=90)
#   st.pyplot(fig)

# #------------------BoxPlot--------------------
# if st.checkbox('Построить график распределения вероятностей "Ящик с усами" (BoxPlot)'): #Boxplot (Ящик с усами) — это график, отражающий форму распределение, медиану, квартили и выбросы.
#   st.write('*BoxPlot* - показывает медиану (линия внутри ящика), нижний (25%) и верхний квартили(75%), минимальное и максимальное значение выборки (усы) и ее выбросы')
#   # image = Image.open('boxplot.png')
#   # st.image(image)
#   fig, ax = plt.subplots() 
#   fig = plt.figure(figsize=(20,10))
#   plt.xticks(rotation=45)
#   plt.ticklabel_format(style='plain')
#   ax_x = st.multiselect('Ось Х (выберите один признак)', df_ohe.columns.tolist())
#   ax_y = st.multiselect('Ось У (выберите один признак)', df_ohe.columns.tolist())
#   if not ax_x or not ax_y:
#     st.write('*Признаки для посторения графика не выбраны*')
#   else:
#     ax = sns.boxplot(x=df_ohe[ax_x[0]], y=df_ohe[ax_y[0]])
#     st.pyplot(fig)

st.markdown('''<h2 style="text-align: left; color: black;">Блок 3: Прогноз оттока клиентов с помощью инструментов машинного обучения: </h2>''', 
            unsafe_allow_html=True)
st.markdown('''<h3 style="text-align: left; color: black;">Часть 1: Данные без балансировки </h3>''', 
            unsafe_allow_html=True)
st.markdown(''' \nЧтобы прогноз оттока клиентов был максимально точным и своевременным, эту задачу стоит поручить не людям, а алгоритмам машинного обучения (Machine Learning). Давайте разберёмся, как это работает.
\n Перед нами наши данные (Рисунок 1): ряд признаков (колонок) по каждому клиенту (строке) в результате дают итог: останется этот клиент в банке (значение в колонке "Exited"=0) или уйдёт от нас (значение в колонке "Exited"=1)
''')
img_datatable = Image.open('raw_data_arrow.png') #
st.image(img_datatable, use_column_width='auto', caption= 'Рисунок 1: вся совокупность признаков прогнозирует целевую переменную') # width=450
st.markdown(''' И чтобы найти верную зависимость "признаки -> целевая переменная" используются различные алгоритмы машинного обучения.
\n В нашем примере мы будем использовать **GradientBoostingClassifier** - одну из моделей библиотеки scikit-learn, которая обладает высокой точностью и, поэтому, часто используется в реальных бизнес-задачах.
\n Чтобы модель показывала точные результаты, необходимо в начале настроить её гиперпараметры: learning rate, max_depth, n_estimators, subsample.
\n **Что значат гиперпараметры?**
\nЭто параметры, значения которых задается до начала обучения модели и не изменяется в процессе обучения. Гиперпараметры используются для управления процессом обучения.
\nlearning rate - коэффициент скорости обучения,
\nmax_depth - глубина обучающего дерева,
\nn_estimators - количество обучающих деревьев
\nsubsample -  то, какими долями данные будут посылаться в модель
\n **Оценивать точность модели мы будем по 2 метрикам: F1-score и ROC-AUC:**
\n F1-score представляет собой гармоническое среднее между точностью и полнотой классификационной модели. Принимает значение от 0 д 1. Чем ближе к 1, тем качественнее наш классификатор.
\n ROC-AUC - это площадь под кривой ROC на графике зависимости между чувствительностью классификационной модели и её специфичностью. Принимает значение от 0 д 1. И также, чем больше значение ROC-AUC - тем лучше модель.
\n*Задание по Блоку 3 Части 1:*
\n 1. Подберите такую комбинацию гиперпараметров модели GradientBoostingClassifier, чтобы F1-score был не менее 0.55 и ROC-AUC не менее 0.70
\n 2. Запишите найденные гиперпараметры в Бланк лабораторной работы.
\n Если задания показались сложными - воспользуйтесь подсказкой ниже:''')
if st.button('Показать подсказку', key='hint2'):
  st.write('''Практика показала, что для этой задачи оптимальные параметры модели (т.е. которые дают наивысший показатель метрик при минимальных ресурсах и времени обучения) - следующие:
  \n learning_rate: 0.1, max_depth: 5, n_estimators: 100, subsample: 0.8''')

#------------------cycle_fit_predict():--------------------
def cycle_fit_predict(df, lr, maxd, nest, ssampl):
  # выделим таргет и фичи: 
  target = df['Exited']
  features = df.drop('Exited', axis=1)
  # train_test_split:
  X_train, X_test, y_train, y_test = train_test_split(features, 
                                                      target, 
                                                      test_size=0.2,
                                                      random_state=42)  
  # scaler:
  scaler.fit(X_train[numeric])
  X_train[numeric] = scaler.transform(X_train[numeric])
  X_test[numeric] = scaler.transform(X_test[numeric])
  # fit-predict:
  model = GradientBoostingClassifier(learning_rate=lr, max_depth=maxd, n_estimators=nest, subsample=ssampl, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred)
  return f1, roc_auc, X_test, y_test, y_pred, model

#------------------Выбор гиперпараметров модели БЕЗ БАЛАНСИРОВКИ КЛАССОВ:--------------------
student_learning_rate_1 = st.multiselect('Выберите коэффициент скорости обучения (learning rate):', [0.1, 0.05, 0.01], key='student_learning_rate_1')
student_max_depth_1 = st.multiselect('Выберите глубину обучающего дерева (max_depth):', [3, 5, 10], key='student_max_depth_1')
student_n_estimators_1 = st.multiselect('Выберите количество обучающих деревьев (n_estimators):', [100, 130, 150], key='student_n_estimators_1')
student_subsample_1 = st.multiselect('Выберите, какими долями данные будут посылаться в модель (subsample):', [0.6, 0.8, 1.0], key='student_subsample_1')

if st.checkbox('Начать обучение'):
  if not student_learning_rate_1 or not student_max_depth_1 or not student_n_estimators_1 or not student_subsample_1:
    st.write('*Выбраны не все гиперпараметры модели*')
  else:
    student_f1_1, student_roc_auc_1, X_test_1, y_test_1, y_pred_1, model_1  = cycle_fit_predict(df_ohe, lr=student_learning_rate_1[0], maxd=student_max_depth_1[0], nest=student_n_estimators_1[0], ssampl=student_subsample_1[0])
    st.write('Показатель F1-score:', round(student_f1_1, 2),  'и значение ROC-AUC:', round(student_roc_auc_1, 2))

#------------------Выбор гиперпараметров модели С БАЛАНСИРОВКОЙ КЛАССОВ:--------------------
st.markdown('''<h3 style="text-align: left; color: black;">Часть 2: С балансировкой данных </h3>''', 
            unsafe_allow_html=True)
st.markdown('''Но результаты нашей модели стоит улучшить (см.Рисунок 2)''')
img_criterion = Image.open('ROC-кривая_критерии.jpg') #
st.image(img_criterion, use_column_width='auto', caption= 'Рисунок 2: традиционная экспертная шкала оценки качества модели по площади под ROC-кривой') # width=450
st.markdown('''Давайте посмотрим, почему у модели могут быть не самые лучшие метрики.
\n Одна из причин - дисбаланс классов целевой переменной в данных, на которых мы обучаем нашу модель. **Приведём пример**: 
если выборка наших клиентов такова, что в ней 90% клиентов будут уходить, 
а 10% оставаться - то с большей вероятностью, модель, обученная на таких данных будет прогнозировать уход клиента из банка.
\n Проверьте, есть ли дисбаланс по целевой переменной среди ваших клиентов?''')

if st.checkbox('Проверить дисбаланс в исходных данных'): 
  # st.write(df_ohe['Exited'].value_counts()) 
  fig = plt.figure(figsize=(20,10))
  plt.xticks(rotation=0)
  plt.ticklabel_format(style='plain')
  sns.barplot(data=df_ohe, x=df_ohe['Exited'].value_counts().index, y=df_ohe['Exited'].value_counts())
  plt.xlabel('Клиент остался/ушёл')
  plt.ylabel('Количество клиентов')
  plt.rc('font', size=20)  
  # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
  # plt.rc('ytick', labelsize=20) 
  st.pyplot(fig)

  st.write(f'''**Оставшихся клиентов:** {df_ohe['Exited'].value_counts()[0]}, их доля в общем количестве клиентов: {df_ohe['Exited'].value_counts()[0]*100/(df_ohe['Exited'].value_counts()[0]+df_ohe['Exited'].value_counts()[1])}%
  \n **Ушедших клиентов:** {df_ohe['Exited'].value_counts()[1]}, их доля в общем количестве клиентов: {df_ohe['Exited'].value_counts()[1]*100/(df_ohe['Exited'].value_counts()[0]+df_ohe['Exited'].value_counts()[1])}%
  \n **Доля меньшего класса относительно большего:** {round(df_ohe['Exited'].value_counts()[1]*100/df_ohe['Exited'].value_counts()[0], 2)}%''')

#------------------downsample():--------------------
def downsample(dataframe, column, fraction): 
    clients_zeroes = dataframe[dataframe[str(column)]==0]
    clients_ones = dataframe[dataframe[str(column)]==1]
    df_downsamped = pd.concat([clients_zeroes.sample(frac=fraction, random_state=42)] + [clients_ones]) # frac - fraction of axis items to RETURN
    df_downsamped = shuffle(df_downsamped, random_state=42)
    return df_downsamped

st.markdown('''Как видно, выборка клиентов не сбалансирована: одного класса целевой переменной в 4 раза больше, чем другого. 
Обучать модель на таких данных не совсем корректно, нужна балансировка классов.
\nОдним из методов балансировки является **даунсемплирование (downsamping)**: частичное "урезание" слишком распространённого класса.
\n*Задание по Блоку 3 Части 2:*
\n1. Чтобы классы целевой переменной были более сбалансирваны, введите долю оставшихся клиентов банка, которую мы оставим в обучающей выборе (остальные строки по оставшимся клиентам использовать не будем).
\n2. Оцените по графику соотношение оставшихся/ушедших клиентов после балансировки классов.
\n3. Обучите модель на новой, сбалансированной выборке.
\n4. Запишите найденные лучшие гиперпараметры для модели после даунсемплинга в Бланк лабораторной работы.
\n5. Посмотрите точность 2х моделей на графике. Оцените визуально качество предсказаний без балансировки классов и после даунсемплинга.
\n6. Запишите выводы в Бланк лабораторной работы.
\nЕсли задания показались сложными - воспользуйтесь подсказкой ниже:''')
if st.button('Показать подсказку', key='hint3'):
  st.write('''Практика показала, что если долю большего класса целевой переменной ("Exited=0") приводить к доле меньшего класса ("Exited=1"), 
  т.е. выбирать fraction=0.25, то оптимальный показатель метрик дают следующие гиперпараметры:
  learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8''')

student_fraction_2 = st.multiselect('Выберите долю наибольшего класса (fraction), что мы передадим в обучение модели:', [0.5, 0.35, 0.25, 0.15], key='student_fraction_2')

if st.checkbox('Проверить соотношение сбалансированных данных'): 
  if not student_fraction_2:
    st.write('*Вы не произвели балансировку классов целевой переменной*')
  else:
    df_downsampled = downsample(df_ohe, 'Exited', student_fraction_2[0])
    fig = plt.figure(figsize=(20,10))
    plt.xticks(rotation=0)
    plt.ticklabel_format(style='plain')
    sns.barplot(data=df_downsampled, x=df_downsampled['Exited'].value_counts().index, y=df_downsampled['Exited'].value_counts())
    plt.xlabel('Выборка после даунсемплинга: клиент остался/ушёл')
    plt.ylabel('Количество клиентов в выборке после даунсемплинга')
    plt.rc('font', size=20) 
    st.pyplot(fig)

student_learning_rate_2 = st.multiselect('Выберите коэффициент скорости обучения (learning rate):', [0.1, 0.05, 0.01], key='student_learning_rate_2')
student_max_depth_2 = st.multiselect('Выберите глубину обучающего дерева (max_depth):', [3, 5, 10], key='student_max_depth_2')
student_n_estimators_2 = st.multiselect('Выберите количество обучающих деревьев (n_estimators):', [100, 150, 200], key='student_n_estimators_2')
student_subsample_2 = st.multiselect('Выберите, какими долями данные будут посылаться в модель (subsample):', [0.6, 0.8, 1.0], key='student_subsample_2')

if st.checkbox('Начать новое обучение модели по сбалансированной выборке'):
  if not student_fraction_2 or not student_learning_rate_2 or not student_max_depth_2 or not student_n_estimators_2 or not student_subsample_2:
    st.write('*Выбраны не все параметры для нового обучения модели*')
  else:
    df_downsampled = downsample(df_ohe, 'Exited', student_fraction_2[0])
    student_f1_2, student_roc_auc_2, X_test_2, y_test_2, y_pred_2, model_2 = cycle_fit_predict(df_downsampled, 
                                                        lr=student_learning_rate_2[0], 
                                                        maxd=student_max_depth_2[0], 
                                                        nest=student_n_estimators_2[0], 
                                                        ssampl=student_subsample_2[0])
    st.write('Показатель F1-score:', round(student_f1_2, 2),  'и значение ROC-AUC:', round(student_roc_auc_2, 2))

#------------------ROC-AUC:--------------------
if st.checkbox('Посмотреть точность 2х моделей на графике'): 
  try:
    probabilities_baseline_gbc = model_1.predict_proba(X_test_1)
    probabilities_one_baseline_gbc = probabilities_baseline_gbc[:, 1]
    fpr_baseline_gbc, tpr_baseline_gbc, thresholds_baseline_gbc = roc_curve(y_test_1, probabilities_one_baseline_gbc)

    probabilities_gbc_downs = model_2.predict_proba(X_test_1)
    probabilities_one_gbc_downs = probabilities_gbc_downs[:,1]
    fpr_gbc_downs,tpr_gbc_downs,thresholds_gbc_downs = roc_curve(y_test_1, probabilities_one_gbc_downs)

    plt.figure()
    fig = plt.figure(figsize=(20,14))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr_baseline_gbc, tpr_baseline_gbc)
    plt.plot(fpr_gbc_downs,tpr_gbc_downs)
    # plt.plot(fpr_gbc_ups,tpr_gbc_ups)
    plt.xlabel('Специфичность классификации моделью')
    plt.ylabel('Чувствительность классификации моделью')
    plt.title('ROC-кривая')
    plt.legend(('Случайное предсказание','Модель без балансировки классов','Модель после даунсемплинга'),
              loc= 'lower right') 
    plt.rc('font', size=20) 
    st.pyplot(fig)

    st.write('''*Как понимать график ROC-кривой?*
    \nКак было сказано выше, график ROC-кривой описывает взаимосвязь между чувствительностью модели и её специфичностью при классификации наших клиентов.
    \nПлощадь под ROC-кривой отражает показатель ROC-AUC: чем ближе он к единице (потому что вся площадь координатной сетки равна 1.0), тем лучше качество модели (см. Рисунок 2)
    \nПоэтому по данному графику легко понять: чем большую площадь квадрата отсекает ROC-кривая модели - тем лучше у неё предсказания.''')

    result_scores = {'Модели': ['Модель без балансировки классов', 'Модель после даунсемплинга'], 
                  'F1-score': [f1_score(y_test_1, y_pred_1), f1_score(y_test_2, y_pred_2)],
                  'ROC-AUC': [roc_auc_score(y_test_1, y_pred_1), roc_auc_score(y_test_2, y_pred_2)]}

    result_table_scores = pd.DataFrame(result_scores)  
    st.write('*Итоговая таблица метрик 2х моделей:*')    
    st.table(result_table_scores) 
  except:
    st.write('*Вначале обучите одну из моделей*')

st.markdown('''<h3 style="text-align: left; color: black;">Часть 3: Предсказание лучшей моделью </h3>''', 
            unsafe_allow_html=True)
st.markdown('''\n*Задание по Блоку 3 Части 3:*
\n 1. Выберите характеристика клиента, про которого хотите узнать: останется ли он в банке или покинет его.
\n 2. Запишите предскзанный результат в Бланк лабораторной работы.''')

student_CreditScore	= st.multiselect('Выберите кредитный рейтинг клиента', [i for i in range(df_ohe['CreditScore'].min(), df_ohe['CreditScore'].max()+50, 50)])
student_Age = st.number_input('Введите возраст клиента', min_value=18, max_value=100)
student_Tenure = st.multiselect('Выберите срок пребывания клиентом банка', [i for i in range(0, 11, 1)])
student_Balance = st.number_input('Введите баланс на счёте клиента', min_value=0, max_value=261000)
student_NumOfProducts = st.multiselect('Выберите количество банковских продуктов', [i for i in range(df_ohe['NumOfProducts'].min(), df_ohe['NumOfProducts'].max()+1, 1)])
student_HasCrCard = st.multiselect('Является ли держателем кредитной карты?', ['Да', 'Нет'])
student_IsActiveMember = st.multiselect('Активно пользуется услугами банка?', ['Да', 'Нет'])
student_EstimatedSalary = st.number_input('Введите доход клиента', min_value=0, max_value=20000)
student_Geography = st.multiselect('Выберите город проживания клиента', ['Москва', 'Санкт-Петербург', 'Казань'])
student_Gender = st.multiselect('Выберите пол клиента', ['Мужской', 'Женский'])

# st.write(X_test_1)
if not student_CreditScore or not student_Age or not student_Tenure or not student_Balance or not student_NumOfProducts or not student_HasCrCard or not student_IsActiveMember or not student_EstimatedSalary or not student_Geography or not student_Gender:
  st.write('*Выбраны не все параметры для предсказания моделью*')
else:
  student_choise = {'CreditScore': student_CreditScore[0], 
                'Age': float(student_Age),
                'Tenure': float(student_Tenure[0]),
                'Balance': float(student_Balance),
                'NumOfProducts':student_NumOfProducts[0],
                'HasCrCard': 1 if student_HasCrCard[0]=='Да' else 0,
                'IsActiveMember': 1 if student_IsActiveMember[0]=='Да' else 0,
                'EstimatedSalary': float(student_EstimatedSalary),
                'Geography_Kazan': 1 if student_Geography[0]=='Казань' else 0,
                'Geography_Moscow':1 if student_Geography[0]=='Москва' else 0,
                'Geography_Saint_Petersburg': 1 if student_Geography[0]=='Санкт-Петербург' else 0,
                'Gender_Female': 1 if student_Gender[0]=='Женский' else 0,
                'Gender_Male': 1 if student_Gender[0]=='Мужской' else 0}

  table_student_choise = pd.DataFrame(student_choise, index=[0])
  st.write('Таким образом клиент имеет следующие характеристики:')
  table_student_choise

if st.checkbox('Посмотреть предсказания лучшей модели по клиенту'):
  model_loaded = pickle.load(open("gbc_downs.pkl", "rb"))
  student_pred = model_loaded.predict(table_student_choise)
  # st.write(student_pred)
  if student_pred[0]==1: 
    st.markdown('''<h3 style='text-align: left; color: red;'>Вывод: клиент в ближайшее время уйдет</h3>''', unsafe_allow_html=True)
  else:
    st.markdown('''<h3 style='text-align: left; color: green;'>Вывод: клиент останется в банке</h3>''', unsafe_allow_html=True)

st.markdown('''<h2 style="text-align: left; color: black;"> Выводы по второй части лабораторной работы: </h2>''', 
            unsafe_allow_html=True)
st.markdown('''
\n 1. Методы машинного обучения активно применяются для прогнозирования задач бизнеса в разных сферах.
\n 2. Одна из наиболее популярных библиотек для данных целей - [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier).
\n 3. Практически никогда не реализуют точный прогноз с первого раза: вначале используют базовую модель, 
а результаты последующих моделей сравнивают с результатами baseline-модели. Естественно, лучшей моделью считается та, что даёт более высокие метрики качества.''')
