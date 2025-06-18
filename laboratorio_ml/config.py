# Parámetros para RandomizedSearchCV
parametros = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'multi_class': ['multinomial']
}

# Parámetros para el modelo
model_params = {
    'max_iter': 1000
}

# Parámetros para la división de datos
test_size = 0.33
random_state = 42
