# charge les chemins vers les fichiers de données : base_processed, base_raw, base_models...
# Read the contents of the file
with open('init_notebook.py') as f:
    code = f.read()

# Execute the code
exec(code)


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données depuis les fichiers CSV
X_train_scaled = pd.read_csv(base_processed + 'X_train_scaled.csv')
X_test_scaled = pd.read_csv(base_processed + 'X_test_scaled.csv')
y_train = pd.read_csv(base_processed + 'y_train.csv')
y_test = pd.read_csv(base_processed + 'y_test.csv')

X_train_scaled=X_train_scaled.replace({False: 0, True: 1})
X_test_scaled=X_test_scaled.replace({False: 0, True: 1})

# on formate la variable cible
y_column = "Ewltp (g/km)"
y_train = y_train[y_column]
y_test = y_test[y_column]

print("head of y_test for info : ")
print(y_test.head())

# Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Évaluer la performance du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")


# validation croisée
from sklearn.model_selection import cross_val_score

model = LinearRegression()
scores = cross_val_score(model, X_train_scaled, y_train, cv=8, scoring='r2')
print("Cross-validated R^2 scores:", scores)
print("Average R^2:", scores.mean())


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# En cas d'exécution indépendante des travaux précédents: Charger les données depuis les fichiers CSV
#from sklearn.metrics import mean_squared_error, r2_score
#X_train_scaled = pd.read_csv(base_processed + 'X_train_scaled.csv')
#X_test_scaled = pd.read_csv(base_processed + 'X_test_scaled.csv')
#y_train = pd.read_csv(base_processed + 'y_train.csv')
#y_test = pd.read_csv(base_processed + 'y_test.csv')
#X_train_scaled = X_train_scaled.replace({False: 0, True: 1})
#X_test_scaled = X_test_scaled.replace({False: 0, True: 1})
# s'assurer que la variable cible est correctement formatée
#y_column = "Ewltp (g/km)"
#y_train = y_train[y_column]
#y_test = y_test[y_column]

# Modèle de regression Ridge
ridge = Ridge()

# Grille des hyperparamètres à tester
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

# Recherche par grille
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Affichage des meilleurs hyperparamètres
print("Meilleurs hyperparameters:", grid_search.best_params_)
print("Meilleur R^2 score:", grid_search.best_score_)

# Utiliser le meilleur modèle pour prédire sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Performance du modèle sur l'ensemble de test
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE on test set: {mse}")
print(f"R^2 on test set: {r2}")

# Afficher les coefficients du modèle
coefficients = pd.DataFrame(best_model.coef_, X_train_scaled.columns, columns=['Coefficient'])
print(coefficients)





param_grid = {
    'alpha': [0.01, 0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.25, 0.3, 0.38, 0.4, 0.42, 0.44, 0.5, 0.6, 1.0, 1.1, 5.0, 10.0, 100.0]
}

# Recherche par grille
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Affichage des meilleurs hyperparamètres
print("Meilleurs hyperparameters:", grid_search.best_params_)
print("Meilleur R^2 score:", grid_search.best_score_)

# Utiliser le meilleur modèle pour prédire sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Performance du modèle sur l'ensemble de test
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE on test set: {mse}")
print(f"R^2 on test set: {r2}")

# Afficher les coefficients du modèle
coefficients = pd.DataFrame(best_model.coef_, X_train_scaled.columns, columns=['Coefficient'])
print(coefficients)






import time
import pandas as pd
import cuml
from cuml.preprocessing import PolynomialFeatures
from cuml.feature_selection import SelectKBest, f_regression
from cuml.preprocessing import StandardScaler
from cuml.linear_model import Ridge
from cuml.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import cudf

# Convert pandas DataFrames to cuDF DataFrames
X_train_scaled = cudf.DataFrame.from_pandas(X_train_scaled)
X_test_scaled = cudf.DataFrame.from_pandas(X_test_scaled)
y_train = cudf.Series(y_train)
y_test = cudf.Series(y_test)

# Ajouter des caractéristiques polynomiales
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Sélection des meilleures caractéristiques
selector = SelectKBest(score_func=f_regression, k='all')
X_train_best = selector.fit_transform(X_train_poly, y_train)
X_test_best = selector.transform(X_test_poly)

# Définir le pipeline de standardisation et de modélisation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Définir la grille des hyperparamètres à tester
param_grid = {  
    'ridge__alpha': [0.5, 1, 2.0, 4.0],  
    'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'],  
    'ridge__max_iter': [150, 1500]  
}  

# Mettre en place la recherche par grille
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)

start_time = time.time()
# Effectuer la recherche par grille
grid_search.fit(X_train_best, y_train)

# Afficher les meilleurs hyperparamètres et le score correspondant
print("Meilleurs hyperparamètres (jeu d'entraînement):", grid_search.best_params_)
print("Meilleur R^2 score (jeu d'entraînement):", grid_search.best_score_)

# Utiliser le meilleur modèle pour prédire sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_best)

# Évaluer la performance du modèle sur l'ensemble de test
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test.to_pandas(), y_pred.to_pandas())
r2 = r2_score(y_test.to_pandas(), y_pred.to_pandas())

print(f"Mean Squared Error sur jeu de test: {mse}")
print(f"R^2 sur jeu de test: {r2}")

# Access the Ridge model within the pipeline to retrieve the coefficients
ridge_model = best_model.named_steps['ridge']

# Get the coefficients
coefficients = pd.DataFrame(ridge_model.coef_.to_pandas(), poly.get_feature_names_out(X_train_scaled.columns), columns=['Coefficient'])
print(coefficients)

end_time = time.time()
single_iteration_time = end_time - start_time
print(f"Time taken: {single_iteration_time} seconds")







