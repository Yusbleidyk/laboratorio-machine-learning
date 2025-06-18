import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from dataset import load_data, preprocess_data
from features import separate_variables, balance_classes
from plots import plot_confusion, plot_roc
import config

# Cargar y preprocesar datos
data = load_data("data/churn.csv")
data = preprocess_data(data)

# Separar variables y balancear clases
X, y = separate_variables(data)
X_res, y_res = balance_classes(X, y)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=config.test_size, random_state=config.random_state)

# Entrenamiento y optimización de hiperparámetros
model = LogisticRegression(**config.model_params)
search = RandomizedSearchCV(model, config.parametros, n_iter=10, cv=3, scoring='accuracy')
search.fit(X_train, y_train)

# Mejor modelo
better_model = search.best_estimator_

# Guardar el modelo
modelo_path = f"../laboratorio-machine-learning-main/churn/models/better_model.pk"
with open(modelo_path, 'wb') as f:
    pickle.dump(better_model, f)

# Evaluación
y_pred = better_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualización
plot_confusion(better_model, X_test, y_test)
plot_roc(better_model, X_test, y_test)

