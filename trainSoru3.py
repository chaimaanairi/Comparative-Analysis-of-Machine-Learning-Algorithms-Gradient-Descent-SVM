import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from custom_metrics import confusion_matrix_custom, accuracy_score_custom, precision_score_custom, recall_score_custom, f1_score_custom
from dataset import load_data  # Veri kümesini yükleyen fonksiyon

# Veriyi yükle
X_train, y_train, X_val, y_val, X_test, y_test = load_data()

# Kernel parametre arama değerleri
param_grid = {
    'C': [0.1, 1, 10],  # Penaltı parametresi
    'gamma': ['scale', 'auto'],  # RBF kernel için gamma, diğer kernel'ler için degree kullanılabilir
    'degree': [2, 3]  # Polinomsal kernel için degree
}


# SVM Modeli ve GridSearchCV için fonksiyon
def svm_grid_search(kernel_type, param_grid):
    """
        Belirtilen kernel tipi ve parametre ızgarasıyla SVM modeli için en iyi parametreleri GridSearchCV ile arar.

        Args:
            kernel_type (str): SVM kernel tipi ('linear', 'poly', 'rbf').
            param_grid (dict): Hiperparametre arama ızgarası.

        Returns:
            tuple: (En iyi modeli temsil eden SVC nesnesi, en iyi parametrelerin sözlüğü).
    """
    if kernel_type == 'linear':
        model = svm.SVC(kernel='linear')
    elif kernel_type == 'poly':
        model = svm.SVC(kernel='poly')
    elif kernel_type == 'rbf':
        model = svm.SVC(kernel='rbf')

    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


# Kernel tipleri için en iyi parametreleri bul ve model eğit
best_models = {}
best_params = {}

for kernel in ['linear', 'poly', 'rbf']:
    print(f"Grid Search için {kernel} kernel")
    best_model, best_param = svm_grid_search(kernel, param_grid)
    best_models[kernel] = best_model
    best_params[kernel] = best_param
    print(f"En iyi parametreler: {best_param}")


# En başarılı modelin karar sınırlarını çizdirme
def plot_decision_boundary(X, y, model, title):
    """
        Modelin karar sınırlarını çizdirir.

        Args:
            X (numpy.ndarray): Özellik matrisi.
            y (numpy.ndarray): Hedef etiketler.
            model (SVC): Eğitilmiş SVM modeli.
            title (str): Grafiğin başlığı.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()


# Karar sınırlarını çizdir
plt.figure(figsize=(8, 6))
plot_decision_boundary(X_train, y_train, best_models['rbf'], title="En İyi RBF Kernel Karar Sınırı")
plt.savefig("results/svm_rbf_decision_boundary.png")
plt.show()


# Model performansını değerlendirme (confusion matrix ve metrikler)
def evaluate_model(model, X_test, y_test):
    """
        Modelin test setindeki performansını değerlendirir.

        Args:
            model (SVC): Eğitilmiş SVM modeli.
            X_test (numpy.ndarray): Test özellik matrisi.
            y_test (numpy.ndarray): Test hedef etiketleri.

        Prints:
            Accuracy, precision, recall, f1-score.
            Çizim olarak confusion matrix'i gösterir.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix_custom(y_test, y_pred)

    accuracy = accuracy_score_custom(y_test, y_pred)
    precision = precision_score_custom(y_test, y_pred)
    recall = recall_score_custom(y_test, y_pred)
    f1 = f1_score_custom(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(2), ['Negative', 'Positive'])
    plt.yticks(np.arange(2), ['Negative', 'Positive'])
    plt.grid(False)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.savefig(f"results/{model.kernel}_confusion_matrix.png")
    plt.show()


# Her kernel için performans değerlendirmesi
for kernel, model in best_models.items():
    print(f"\n{kernel} kernel için performans:")
    evaluate_model(model, X_test, y_test)
