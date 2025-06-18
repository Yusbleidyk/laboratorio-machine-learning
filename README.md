# laboratorio-machine-learning
API Simple que sirve un modelo para la prevención del Churn

## Tecnologías usadas
This API uses a number of open source projects to work properly:

* [Python] - Lenguaje de programación
* [Flask] - Libreria minimalista para el provisionamiento de APIs
* [scikit-learn] - Libreria de aprendizaje automatico
* [pickle] - Libreria para cargar y guardar variables en memoria

## Instalación Local
Esta API requiere Python y las librerias señaladas en el requirements.txt

1. Clonar repositorio
2. Correr la api, ejecutando desde la carpeta de proyecto

```
python app.py
```
## Cambios recientes y mejoras

- Modularización del código: se crearon `train.py`, `features.py`, `dataset.py`, `config.py`, `plots.py`.
- Reentrenamiento del modelo con `SMOTE` y `RandomizedSearchCV`.
- Visualización de resultados con matriz de confusión y curva ROC.
- Separación de configuración en `config.py`.
- Inclusión del notebook `model_retrain.ipynb` para documentación del proceso.

## Reentrenar el modelo

Para reentrenar el modelo desde consola:

python laboratorio_ml/train.py


Esto generará un nuevo modelo en:

churn/models/better_model.pk


## Estructura del proyecto

LABORATORIO-MACHINE-LEARNING-MAIN/
│
├── churn/
│   └── models/
│       ├── features_retrain.pk
│       └── model.pk
├── data/
│   └── churn.csv
│
├── laboratorio_ml/
│   ├──__init__.py
│   ├── app.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── train.py
│
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── model_retrain.ipynb
│   └── modelgeneration_27092023_ricalanis.ipynb
│
├── LICENSE
├── README.md
└── requirements.txt

#### Verificar que la app esta corriendo exitosamente

La api se verifica mandando un ejemplo para confirmar que esta funcionando correctamente

```
curl --location --request GET 'http://localhost:3001/query?feats=465,France,Female,51,8,122522.32,1,0,0,181297.65'
```

Si todo esta bien, deberías de obtener una respuesta así
```
{"response": [1]}
```

#### Equipo

* Juan Perez Nombrefalso
* Ricardo Alanís Tamez

#### Contribuir

Hacer fork del repositorio y mandar pull request con los cambios propuestos

