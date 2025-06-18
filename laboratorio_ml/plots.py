import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay


def plot_confusion(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.show()

def plot_roc(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("El modelo no tiene el método predict_proba necesario para la curva ROC.")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_disp.plot()
    plt.title('Curva ROC')
    plt.show()
