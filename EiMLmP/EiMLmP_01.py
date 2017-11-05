# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:24:42 2017

@author: di30409
"""
#Importiere Module
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Lade Iris Daten für Machine Learning Test und gib Ihalte aus
iris_dataset = load_iris()
print("DESCR:\n{0}...".format(iris_dataset['DESCR'][:100]))
print("feature_names:\n{0}".format(iris_dataset['feature_names']))
print("target_names:\n{0}".format(iris_dataset['target_names']))
print("data:\n{0}...".format(iris_dataset['data'][:20]))
print("target:\n{0}...".format(iris_dataset['target'][:20]))

#Aufsplitten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#Sichten der Daten durch ein Wertepaarediagramm (todo: MGLEARN importieren!)
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8)

#Erstelle neue K-Nachbarn-Klassifizierer Objekt und trainiere
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

#Prüfe Verallgemeinerung durch manuelle Testdaten für Vorhersage der Spezies
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Vorhersage:\n{0}".format(prediction))
print("Vorhergesagter Name:\n{0}".format(iris_dataset['target_names'][prediction]))

#Prüfe Verallgemeinerung durch "automatische" Testdaten für Vorhersage der Spezies
prediction = knn.predict(X_test)
print("Vorhersage:\n{0}".format(prediction))
print("Vorhersage der Testdaten:\n{0}".format(y_test))

#Prüfe Genauigkeit der Vorhersagen im Vergleich zu Testdaten
#print("Genauigkeit Vorhersagen zu Testdaten:\n{0:.2f}".format(np.mean(prediction == y_test)))
print("Genauigkeit Vorhersagen zu Testdaten:\n{0:.2f}".format(knn.score(X_test,y_test)))