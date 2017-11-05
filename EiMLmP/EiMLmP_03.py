# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Standardmodule für Tests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython import display
from sklearn.model_selection import train_test_split

#Module für ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Hole Daten
X, y = mglearn.datasets.make_forge()

#Zeichne Daten
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Kategorie 0", "Kategorie1"], loc=4)
plt.xlabel("Erstes Merkmal")
plt.ylabel("Zweites Merkmal")

#Splitte in Trainings- und Testdaten auf
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Erzeuge Algorithmen und trainiere
LogReg = LogisticRegression().fit(X_train, y_train)
LinSVC = LinearSVC().fit(X_train, y_train)
KNC = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
DTC = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
RFC = RandomForestClassifier(n_estimators = 100, random_state=0).fit(X_train, y_train)
GBC = GradientBoostingClassifier(random_state=0, learning_rate=0.01, max_depth=1).fit(X_train, y_train)


#Auswerten der Genauigkeit
print("LogisticRegression:")
print("Score auf den Trainingsdaten: {:.3f}%".format(LogReg.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(LogReg.score(X_test, y_test) * 100))

print("\n\nLinearSVC:")
print("Score auf den Trainingsdaten: {:.3f}%".format(LinSVC.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(LinSVC.score(X_test, y_test) * 100))

print("\n\nK-Neighbors Classifier:")
print("Score auf den Trainingsdaten: {:.3f}%".format(KNC.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(KNC.score(X_test, y_test) * 100))

print("\n\nDecision Tree Classifier:")
print("Score auf den Trainingsdaten: {:.3f}%".format(DTC.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(DTC.score(X_test, y_test) * 100))

print("\n\nRandom Forest Classifier:")
print("Score auf den Trainingsdaten: {:.3f}%".format(RFC.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(RFC.score(X_test, y_test) * 100))

print("\n\nGradient Boosting Classifier:")
print("Score auf den Trainingsdaten: {:.3f}%".format(GBC.score(X_train, y_train) * 100))
print("Score auf den Trainingsdaten: {:.3f}%".format(GBC.score(X_test, y_test) * 100))