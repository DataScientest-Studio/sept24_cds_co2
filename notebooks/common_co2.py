# Pour une exécution indépendante des travaux sur chaque modèle: Charger les données depuis les fichiers CSV

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option('future.no_silent_downcasting', True)

# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
from init_notebook import base_processed, base_raw, base_models

def load_our_data_cat():

    X_train_scaled = pd.read_csv(base_processed + 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(base_processed + 'X_test_scaled.csv')
    y_train = pd.read_csv(base_processed + 'y_train_cat.csv')
    y_test = pd.read_csv(base_processed + 'y_test_cat.csv')
    X_train_scaled = X_train_scaled.replace({False: 0, True: 1}).astype(float)
    X_test_scaled = X_test_scaled.replace({False: 0, True: 1}).astype(float)
    #
    y_column = "categorie"
    y_train = y_train[y_column]
    y_test = y_test[y_column]
    return X_train_scaled, X_test_scaled, y_train, y_test


""" 
Représentation par disques
import matplotlib.pyplot as plt

import pandas as pd

# Combiner y_test et y_pred dans un DataFrame pour calculer les fréquences
data = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
data['count'] = data.groupby(['y_test', 'y_pred'])['y_test'].transform('count')

# Taille des points proportionnelle à la fréquence
sizes = data['count']

# Tracer le graphique
plt.figure(figsize=(8, 8))
plt.scatter(data['y_test'], data['y_pred'], s=sizes, alpha=0.7, label='Données')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y = x')
plt.title("Graphique de Dispersion, après égalisation des classes en set d'entraînement")
plt.xlabel("Valeurs Réelles")
plt.ylabel("Valeurs Prédites")
plt.grid(True)
plt.legend()
plt.show()
 """


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def display_norm_matrix(name, y_predictions, y_test, hyperparams):
    """
    Display the normalized confusion matrix with annotations.
    
    Parameters:
    - name: nom du modèle, pour le titre
    - y_pred: les predictions
    - y_test: array-like, true class labels
    - hyperparams : précisions à apporter au dessus de la matrice
    
    Returns:
    None
    """
    print(f"\n🔹 Matrice de confusion pour {name} and {hyperparams}🔹")
     
    # Confusion matrix
    cm = confusion_matrix(y_test, y_predictions)
    
    # Normalize confusion matrix by class totals
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Display the matrix
    plt.figure(figsize=(8, 6))
    
    # Heatmap with normalized values
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    
    # Overlaying raw values as annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, f"{cm_normalized[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=10)
    
    # Labels and title
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title(f"Matrice de confusion normalisée, {name}")
    plt.show()


# fonction pour courbes ROC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def display_roc(X_test_scaled, y_test, y_pred_proba, model):
    
    # Binarisation des labels (OvR, One-vs-Rest)
    y_test_binarized = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6, 7])

    # Obtenir les probabilités prédites pour chaque classe
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Initialiser le graphique
    plt.figure(figsize=(10, 8))

    # Tracer la courbe ROC pour chaque classe
    for i in range(7):  # Nombre de classes
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Classe {i+1} (AUC = {roc_auc:.2f})')

    # Ligne diagonale (aléatoire)
    plt.plot([0, 1], [0, 1], 'k--', label="Aléatoire")

    # Personnalisation du graphique
    plt.title("Courbes ROC pour chaque classe (One-vs-Rest)")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
