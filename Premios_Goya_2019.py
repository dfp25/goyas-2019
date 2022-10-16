#!/usr/bin/env python
# coding: utf-8

# # Premios Goya

# ## Datos Premios Goya

# In[1]:


################
# Premios Goya #
################

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re


def datos1():
    """
    Primer bloque de datos:
    --> Título de la película.
    --> Nº de nominaciones.
    --> Ganador (1/0).
    Cada lista contiene los datos de una película.
    """
    tp = [i.text for i in html1.find(id='best_picture').parent.findAll('a', 'movie-title-link')]
    np = []
    nomi = html1.find(id='best_picture').parent.select('div.aw-mc')
    for i in nomi:
        if len(i.select('b')) == 0:
            np.append(0)
        else:
            np.append(i.select('b')[0].text)
    gdor = [1] + [0] * (len(tp) - 1)  # Lista con ganador (1) o no (0).
    return tp, np, gdor


def datos2():
    """
    Segundo bloque de datos:
    --> Puntuación media.
    --> Nº de votos recibidos.
    --> Nº de críticas recibidas.
    --> Coproducción (1/0).
    --> Género (lista de géneros principales).
    Cada lista contiene datos homogéneos.
    """
    puntuacion = driver.find_element_by_css_selector('#movie-rat-avg').text.replace(",", ".")
    votos = driver.find_element_by_css_selector('#movie-count-rat > span').text.replace(".", "")
    criticas = driver.find_element_by_css_selector('#movie-reviews-box').text.split()[0]
    coproduccion = [1 if "Coproducción" in i.text else 0 for i in html2.select('dd div.credits span.nb span')]
    if 1 in coproduccion:
        coproduccion = 1
    else:
        coproduccion = 0
    genero = [i.text for i in html2.select('dd span[itemprop="genre"]')]  # Lista de géneros principales.
    return [puntuacion, votos, criticas, coproduccion, genero]


# Listas iniciales
ttl, nmn, gnd, yr = [], [], [], []
dts2 = []

# Cargar páginas años
driver = webdriver.Chrome()
for ed in range(1987, 2019):
    driver.get("https://www.filmaffinity.com/es/awards.php?award_id=goya&year=" + str(ed))

# Bloque 1 de datos
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    titulo, nominaciones, ganador = datos1()
    year = [ed] * len(titulo)
    
# Unión de listas del bloque 1
    ttl.extend(titulo)
    nmn.extend(nominaciones)
    gnd.extend(ganador)
    yr.extend(year)

# Bloque 2 de datos
    for ti in titulo:
        ti = ti.split('(')[0]
        link = [i.get('href') for i in html1.findAll('a', 'movie-title-link', href=True, string=re.compile(ti))]
        driver.get("https://www.filmaffinity.com" + link[0])
        html2 = BeautifulSoup(driver.page_source, 'html.parser')
        dts2.append(datos2())

# Unir datos
d = {'ganador': gnd, 'titulo': ttl, 'year': yr, 'nominaciones': nmn}
df1 = pd.DataFrame(d)
df2 = pd.DataFrame(dts2, columns=['puntuacion', 'votos', 'criticas', 'coproduccion', 'genero'])
df = df1.join(df2)

# Cerrar driver
driver.close()

# Guardar tabla de datos
df.to_csv('df_goyas_data.csv', index=False)


# ## Datos Premios Feroz

# In[2]:


#################
# Premios Feroz #
#################

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def datos_feroz():
    """
    Bloque de datos:
    --> Título de la película drama y comedia.
    --> Ganador drama y comedia (1/0).
    """
    tp_drama = [i.text for i in html1.find(id='mejor_pelicula_drama').parent.findAll('a', 'movie-title-link')]
    gdor_drama = ["ganador"] + ["nominado"] * (len(tp_drama) - 1)
    tp_comedia = [i.text for i in html1.find(id='mejor_comedia').parent.findAll('a', 'movie-title-link')]
    gdor_comedia = ["ganador"] + ["nominado"] * (len(tp_comedia) - 1)
    return tp_drama, gdor_drama, tp_comedia, gdor_comedia


# Listas iniciales
drama_feroz, comedia_feroz, gs_drama, gs_comedia = [], [], [], []

# Cargar páginas años
driver = webdriver.Chrome()
for ed in range(2014, 2019):
    driver.get("https://www.filmaffinity.com/es/awards.php?award_id=feroz&year=" + str(ed))

# Datos Premios Feroz
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    dr_feroz, g_dr_feroz, co_feroz, g_co_feroz = datos_feroz()

# Unión de listas Premios Feroz
    drama_feroz.extend(dr_feroz)
    comedia_feroz.extend(co_feroz)
    gs_drama.extend(g_dr_feroz)
    gs_comedia.extend(g_co_feroz)

# Cerrar driver
driver.close()

# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_data.csv')
dra, com = [], []
fedrlow = [t.lower() for t in drama_feroz]
fecolow = [t.lower() for t in comedia_feroz]
for i in df.titulo.tolist():
    if i.lower() in fedrlow:
        dra.extend([gs_drama[drama_feroz.index(i)]])
    else:
        dra.extend(["no_clasificado"])
    if i.lower() in fecolow:
        com.extend([gs_comedia[comedia_feroz.index(i)]])
    else:
        com.extend(["no_clasificado"])

# Unión Premios Feroz a Premios Goya
df['feroz_drama'] = dra
df['feroz_comedia'] = com

# Guardar tabla de datos Goya + Feroz
df.to_csv('df_goyas_feroz_data.csv', index=False)


# ## Datos Premios Forqué

# In[3]:


##################
# Premios Forqué #
##################

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def datos_forque():
    """
    Bloque de datos:
    --> Título de la película.
    --> Ganador (1/0).
    """
    tp = [i.text for i in html1.find(id='best_picture').parent.findAll('a', 'movie-title-link')]
    gdor = ["ganador"] + ["nominado"] * (len(tp) - 1)
    return tp, gdor


# Listas iniciales
ti_forque, ga_forque = [], []

# Cargar páginas años
driver = webdriver.Chrome()
for ed in range(2010, 2019):
    driver.get("https://www.filmaffinity.com/es/awards.php?award_id=forque&year=" + str(ed))

# Datos Premios Feroz
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    tit_fq, gan_fq = datos_forque()

# Unión de listas Premios Feroz
    ti_forque.extend(tit_fq)
    ga_forque.extend(gan_fq)

# Cerrar driver
driver.close()

# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_feroz_data.csv')
fq = []
tfq = [t.lower() for t in ti_forque]
for i in df.titulo.tolist():
    if i.lower() in tfq:
        fq.extend([ga_forque[ti_forque.index(i)]])
    else:
        fq.extend(["no_clasificado"])

# Unión Premios Feroz a Premios Goya
df['forque'] = fq

# Guardar tabla de datos Goya + Feroz
df.to_csv('df_goyas_total.csv', index=False)


# ## Datos Premios CEC

# In[4]:


###############
# Premios CEC #
###############

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re


def datos_cec():
    """
    Bloque de datos:
    --> Título de la película.
    """
    try:
        if len(html1.find('b', string=re.compile('Película')).parent.parent('i')) != 0:
            tp = html1.find('b', string=re.compile('Película')).parent.parent('i')[0].string
        else:
            tp = html1.find('b', string=re.compile('Película')).parent.parent('em')[0].string
    except:
        return [np.nan]
    tp = tp.split(",")[0].split("\n")
    if len(tp) == 1:
        tp = [tp[0].strip()]
        pass
    else:
        tp1 = tp[0].strip()
        tp2 = tp[1].strip()
        tp = [tp1 + " " + tp2]
    return tp


# Lista de ganadores (procedente de la web)
ti_cec_web = []

# Recopilación web
driver = webdriver.Chrome()
for ed in range(1990, 2019):  # El año es de la producción de la película, no el del premio.
    driver.get("http://www.cinecec.com/EDITOR/premios/palmares/" + str(ed) + ".htm")
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    ti_cec_web.extend(datos_cec())
    
# Cerrar driver
driver.close()

# Lista de ganadores (webs - manual)
ti_cec_man = ['También la lluvia', 
              'No habrá paz para los malvados', 
              'Blancanieves', 
              'Vivir es fácil con los ojos cerrados', 
              'La isla mínima', 
              'Truman', 
              'Tarde para la ira', 
              'La librería']

# Sustitución de lista manual en lista web automática
ti_cec_web[-(len(ti_cec_man)+1):] = ti_cec_man

# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_total.csv')
cec = []
tcw = [t.lower() for t in ti_cec_web]
for i in df.titulo.tolist():
    if i.lower() in tcw:  # Todo en minúsculas.
        cec.extend(["ganador"])
    else:
        cec.extend(["no_clasificado"])

# Unión Premios CEC a Premios Goya
df['cec'] = cec

# Guardar tabla de datos Goya + CEC
df.to_csv('df_goyas.csv', index=False)


# ## Analytical Base Table

# In[6]:


import pandas as pd
import pickle


df = pd.read_csv('df_goyas.csv')

# Columna genero de string a lista
df.genero = df.genero.apply(lambda x: [r.strip("[").strip("]").strip(" ").strip("'") for r in x.split(",")])

# Lista de géneros posibles
y = []
[y.extend(i) for i in df.genero]
lgp = list(set(y))

# Dummy géneros
dummy_generos = pd.DataFrame(columns=lgp, index=df.index)
for n, i in zip(df.genero, df.index):
    for j in n:
        if j in lgp:
            dummy_generos.at[i, j] = 1
dummy_generos.fillna(0, inplace=True)

# Dummy Premios Feroz
dummy_fe_dr = pd.get_dummies(df.feroz_drama, prefix='feroz_drama')
dummy_fe_co = pd.get_dummies(df.feroz_comedia, prefix='feroz_comedia')

# Dummy Premios Forqué
dummy_forque = pd.get_dummies(df.forque, prefix='forque')

# Dummy Premios CEC
dummy_cec = pd.get_dummies(df.cec, prefix='cec')

# Unión df y df_dummy
df = pd.concat([df, dummy_fe_dr, dummy_fe_co, dummy_forque, dummy_cec, dummy_generos], axis=1, sort=False)

# Parche manual: una película en los Goyas al menos tiene una nominación
df.at[df[df.titulo == '27 horas'].index[0], 'nominaciones'] = 1

# Guardar ABT de Goyas
df.to_csv('ABT_Goyas.csv', index=False)

# Guardar lista de géneros
with open("lista_generos.txt", "wb") as f:
    pickle.dump(lgp, f)


# ## Variables dummy. Porcentajes por año de variables numéricas

# In[7]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


df = pd.read_csv('ABT_Goyas.csv')
with open("lista_generos.txt", "rb") as f:
    lgp = pickle.load(f)

# Películas por género
for feature in lgp:
    print(feature, ":", df[feature].sum())
    
# Eliminación columnas usadas para generar dummies
df.drop(columns=['genero', 'feroz_drama', 'feroz_comedia', 'forque', 'cec'], inplace=True)

# Fusión de Premios Feroz drama y comedia (pocos datos en comedia)
df['feroz_ganador'] = df.feroz_drama_ganador  # No hay comedias ganadoras coincidentes Goya-Feroz.
df['feroz_nominado'] = df.feroz_drama_nominado + df.feroz_comedia_nominado

# Se eliminan columnas no usadas
df.drop(columns=['feroz_drama_ganador', 
                 'feroz_drama_no_clasificado', 
                 'feroz_drama_nominado', 
                 'feroz_comedia_no_clasificado', 
                 'feroz_comedia_nominado', 
                 'forque_no_clasificado', 
                 'cec_no_clasificado'], inplace=True)

# Muchos géneros (13). Hacemos 4 grupos:
# Grupo más frecuente: Drama
df['G_drama'] = df.Drama
# Grupo de frecuencia media-alta: Comedia, Thriller, Romance e Intriga.
df['G_med_alto'] = np.where((df.Comedia == 1) 
                               | (df.Thriller == 1) 
                               | (df.Romance == 1) 
                               | (df.Intriga == 1), 1, 0)
# Grupo de frecuencia media-baja: Fantástico, Terror, Acción y Aventuras.
df['G_med_bajo'] = np.where((df['Fantástico'] == 1) 
                               | (df.Thriller == 1) 
                               | (df.Romance == 1) 
                               | (df.Intriga == 1), 1, 0)
# Grupo de frecuencia baja: Western, Ciencia ficción, Musical y Cine negro.
df['G_bajo'] = np.where((df.Western == 1) 
                               | (df['Ciencia ficción'] == 1) 
                               | (df.Musical == 1) 
                               | (df['Cine negro'] == 1), 1, 0)

# Eliminación de géneros ya agrupados
df.drop(columns=lgp, inplace=True)

# Control de Nan en premios sin ediciones (parche)
num_tit = df.groupby('year').count()
n_feroz = num_tit[num_tit.index > 2013].sum()['titulo']
n_forque = num_tit[num_tit.index > 2009].sum()['titulo']
n_cec = num_tit[num_tit.index > 1989].sum()['titulo']
feroz_nan =  [0] * (len(df) - n_feroz) + [1] * n_feroz  # 0 = nan, 1 = sí hay dato.
forque_nan = [0] * (len(df) - n_forque) + [1] * n_forque
cec_nan = [0] * (len(df) - n_cec) + [1] * n_cec
df['feroz_nan'] = feroz_nan
df['forque_nan'] = forque_nan
df['cec_nan'] = cec_nan

# Porcentaje anual de variables cuantitativas
t_anual = df.groupby('year')['nominaciones', 'puntuacion', 'votos', 'criticas'].sum()  # Totales.
df.set_index('year', inplace=True)
ptj = df[['nominaciones', 'puntuacion', 'votos', 'criticas']] / t_anual  # Porcentaje.
df[['nominaciones', 'puntuacion', 'votos', 'criticas']] = ptj
df.reset_index(inplace=True)

# Eliminamos columnas que no vamos a usar como variables independientes
df.drop(columns=['year', 'titulo'], inplace=True)

# Matriz final: 125 registros, 18 variables X. "ganador" variable Y.
# Guardar matriz de datos de Premios Goyas
df.to_csv('Datos_Goyas.csv', index=False)


# ## Modelo '2000' búsqueda de parámetros

# In[1]:


import pickle
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Cargar tabla final de variables
df = pd.read_csv('Datos_Goyas.csv')

# Se utilizan datos solo desde el año 2000 (incluido)
df = df[44:]

# Separar variable target y variables predictoras
y = df.ganador
X = df.drop('ganador', axis=1)

# Dividir X e y en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.ganador)

# Comprobar train y test
print("Se comprueba train y test:")
print(len(X_train), len(X_test), len(y_train), len(y_test))
print(y_train.mean(), y_test.mean())

# Estandarización
scaler_X = StandardScaler().fit(X_train)
X_train_std = scaler_X.transform(X_train)
X_test_std = scaler_X.transform(X_test)

# Algoritmo
clf_gb = GradientBoostingClassifier(random_state=123)

# Hiperparámetros
hyperparameters = {'n_estimators': [100, 200, 500, 1000, 1500, 2000, 3000], 
                   'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], 
                   'max_depth': [1, 3, 5, 7, 9]}

# Modelo
modelo = GridSearchCV(clf_gb, hyperparameters, n_jobs=-1, cv=10)

# Ajustar modelo
modelo.fit(X_train_std, y_train)
print("\nBest Train Score:", modelo.best_score_)

# Matriz de confusión. Predicción de categorías (y_test)
pred = modelo.predict(X_test_std)
print("MC sobre test:", "\n", confusion_matrix(pred, y_test))

# Curva ROC. Predicción de probabilidades (y_test)
pred = modelo.predict_proba(X_test_std)
pred = pred[:, 1]  # Se toma sólo la clase positiva
fpr, tpr, threshols = roc_curve(y_test, pred)

# Dibujar la curva ROC
# Valores gráfico
fig = plt.figure(figsize=(8, 8))
plt.title('Receiver Operating Characteristic (ROC)')
# Plot ROC curve
plt.plot(fpr, tpr, label='gradient boosting')
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0, 1], [0, 1], 'k--')
# Axes limits and labels
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()

# AUROC
print("Área bajo la curva ROC", auc(fpr, tpr))

# Parámetros del modelo
print("Parámtros del modelo ajustado:", modelo.best_estimator_)

# Importancia de las variables
print("Importancia sobre 1:\n")
for nom, val in zip(X, modelo.best_estimator_.feature_importances_):
    print(nom, "\t", round(val, 3))

# Guardar el modelo
with open('modelo_goyas_gb.pkl', 'wb') as f:
    pickle.dump(modelo.best_estimator_, f)


# ## Se ajusta el modelo '2000' con train + test. Predicciones año 2019

# In[1]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Datos: solo features relevantes
df = pd.read_csv('Datos_Goyas.csv', 
                 usecols=['ganador', 'nominaciones', 'puntuacion', 
                          'votos', 'criticas', 'cec_ganador', 'forque_ganador'])

# Utilizar solo datos desde el año 2000 (incluido)
df = df[44:]

# Separar variable target y variables predictoras
y = df.ganador
X = df.drop('ganador', axis=1)

# Estandarización de todo X
scaler_X = StandardScaler().fit(X)
X_scaler = scaler_X.transform(X)

# Algoritmo con parámetros del mejor modelo
mod_final_goyas = GradientBoostingClassifier(n_estimators=1500, learning_rate=0.005, max_depth=1, random_state=123)

# Ajuste del modelo sobre todos los datos
mod_final_goyas.fit(X_scaler, y)

# Guardar el modelo '2000' sobre todas las películas
with open('modelo_goyas_gb_todas.pkl', 'wb') as f:
    pickle.dump(mod_final_goyas, f)

# Diagramas de barras con importancia de características
#nom_var = ['Nº Nominaciones', 'Puntuación media', 'Nº Votos','Nº Críticas', 
#           'Coproducción', 'Ganador Premios Forqué', 'Nominado Premios Forqué', 
#           'Ganador Premios CEC', 'Ganador Premios Feroz', 'Nominado Premios Feroz', 
#           'Género Drama', 'Géneros de frecuencia media-alta', 
#           'Géneros de frecuencia media-baja', 'Géneros de frecuencia baja', 
#           'Feroz nan', 'Forqué nan', 'CEC nan']  # Todas las variables.
nom_var = ['Nominaciones', 'Puntuación', 'Votos','Críticas', 
           'P. CEC', 'P. Forqué']  # Solo variables seleccionadas.

# Data frame con valores mayores que 0, y ordenado
features = {'Variable': nom_var, 'Porcentaje': 100 * mod_final_goyas.feature_importances_}
a = pd.DataFrame(features)
b = a[a.Porcentaje > 0].sort_values(by=['Porcentaje'])
b.index = list(range(len(b)))  # Renombrando índice tras ordenar por valores.

# Gráfico features
fig_features = plt.figure(figsize=(9, 9))
colores = ['aqua', 'yellow', 'red', 'green', 'silver', 'olive', 'navy', 'fuchsia', 'maroon']
plt.barh(b.Variable, b.Porcentaje, align='center', alpha=0.6, color=colores)
# Eje x. Título
plt.xlabel('Porcentaje')
plt.title('RELEVANCIA DE VARIABLES (desde 2000)')
# Etiquetas de datos
for i, fila in b.iterrows():
    plt.text(x=fila.Porcentaje - 1.2, y=i, s="{0:.1%}".format(fila.Porcentaje/100), size=16, color='black')
plt.show()
# Salvar gráfico
fig_features.savefig('features importances')

# Matriz de confusión. Predicción de categorías (y_test)
pred = mod_final_goyas.predict(X_scaler)
print("Matriz Confusión sobre total:", "\n", confusion_matrix(pred, y))

# Curva ROC. Predicción de probabilidades (y_test)
pred = mod_final_goyas.predict_proba(X_scaler)
pred = pred[:, 1]  # Se toma sólo la clase positiva
fpr, tpr, threshols = roc_curve(y, pred)

# Dibujar la curva ROC
# Valor AUROC
valor_auroc = auc(fpr, tpr)
# Valores gráfico
fig_roc = plt.figure(figsize=(9, 9))
plt.title('Receiver Operating Characteristic (ROC)')
# Plot ROC curve
plt.plot(fpr, tpr, label="AUROC: " + "{0:.4}".format(valor_auroc))
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0, 1], [0, 1], 'k--')
# Axes limits and labels
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()
# Salvar gráfico
fig_roc.savefig('ROC_curve')

###

# PREDICCIÓN 2019
# Cargar datos para predicción
df_2019 = pd.read_csv('datos_para_prediccion_2019.csv', sep=";")
titulos_2019 = df_2019.titulo
X_2019 = df_2019.drop('titulo', axis=1)

# Variables cuantitativas en porcentaje anual
v_c = ['nominaciones', 'puntuacion', 'votos', 'criticas']
v_d = ['cec_ganador', 'forque_ganador']
X_2019 = pd.concat([X_2019[v_c] / X_2019[v_c].sum(), X_2019[v_d]], axis=1)

# Estandarización
X_2019_std = scaler_X.transform(X_2019)

# Predicción de probabilidades
pred_2019 = mod_final_goyas.predict_proba(X_2019_std)
pred_2019 = pred_2019[:, 1]

# Ponderación probabilidades sobre 100 (lineal)
probs = pred_2019 / pred_2019.sum()

# Mostrar probabilidades ponderadas sobre 100 para cada título
#print("\nPredicciones películas cargadas...\n")
#tamano = 1 + len(titulos_2019.max())  # Longitud del título más largo para luego rellenar con espacios.
#for tit, pre, prob in zip(titulos_2019, pred_2019, probs):
#    print(tit.ljust(tamano), "\t", "{0:.1%}".format(pre), "\t", "{0:.1%}".format(prob))

# Gráfico probabilidades nominados 2019
# Data frame
n_2019 = pd.DataFrame({'titulo': titulos_2019, 'prob_absoluta': pred_2019, 'prob_edicion': probs})
n_2019 = n_2019.sort_values(by=['prob_edicion'], ascending=False)
n_2019.index = list(range(len(n_2019)))
# Gráfico dimensiones y supra-título
fig, ax = plt.subplots(figsize=(11, 9), facecolor='w')
fig.suptitle('Premios Goya - Mejor Película 2019', fontsize=14, fontweight='bold')
clrs = ['aqua', 'yellow', 'red', 'green', 'orange']
plt.bar(n_2019.titulo, n_2019.prob_edicion, align='center', alpha=0.6, color=clrs)
# Eje x. Título
plt.ylabel('Probabilidad de ganar')
plt.title('¿QUIÉN GANARÁ EL GOYA?')
# Etiquetas de datos
for i, fila in n_2019.iterrows():
    plt.text(x=i, y = 0.03, s="{0:.1%}".format(fila.prob_edicion), 
             size=16, horizontalalignment='center', color='black')    
plt.show()
# Salvar gráfico
fig.savefig('Probs_Goya_2019')


# ## Predicciones para todas las películas: modelo '2000'

# In[ ]:


import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Modelo '2000' entrenado con todas la películas
with open("modelo_goyas_gb_todas.pkl", "rb") as f:
    mggt = pickle.load(f)

# Datos: solo features relevantes
df = pd.read_csv('Datos_Goyas.csv', 
                 usecols=['ganador', 'nominaciones', 'puntuacion', 
                          'votos', 'criticas', 'cec_ganador', 'forque_ganador'])

# Separar variable target y variables predictoras
y = df.ganador
X = df.drop('ganador', axis=1)

# Estandarización solo con los datos del modelo '2000'
scaler_X = StandardScaler().fit(X[44:])
X_scaler = scaler_X.transform(X)

# MODELO '2000' APLICADO A TODAS LAS PELÍCULAS
# Probabilidades de ganar el Goya de todas las películas hasta 2018
pred_todas = mggt.predict_proba(X_scaler)
pred_todas = pred_todas[:, 1]

# Data frame con años, títulos y predicciones
tts = pd.read_csv('ABT_Goyas.csv').loc[:, ['year', 'titulo']]
xx = pd.DataFrame(pred_todas, columns=['probabilidad'])
tts = pd.concat([tts, xx], axis=1)

# Agrupamiento y ponderación
t_anual = tts.groupby('year')['probabilidad'].sum()
tts.set_index('year', inplace=True)
probabilidad_anual = tts['probabilidad'] / t_anual
tts['probabilidad_anual'] = probabilidad_anual
tts.reset_index(inplace=True)
tts['ganador'] = y

# Salvar Data frame
tts.to_csv('probs_todas.csv', encoding='utf-8-sig')

