import numpy as np
import matplotlib.pyplot as plt
from custom_metrics import confusion_matrix_custom, accuracy_score_custom, precision_score_custom, recall_score_custom, f1_score_custom
from tensorflow.keras.models import load_model
from dataset import load_data  # Veri kümesini yükleyen fonksiyon

# Verileri yükle
X_train, y_train, X_val, y_val, X_test, y_test = load_data()

# Eğitilmiş modeli yükle
model = load_model('models/sgd_model.keras')  # Modelin kaydedildiği yolu belirtin

# Model ile tahmin yap
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Tahminleri ikili sınıflara dönüştür

# Performans metriklerini hesapla
def evaluate_model(y_test, y_pred):
    """
    Modelin performans metriklerini hesaplar ve sonuçları ekrana yazdırır.

    Args:
        y_test (numpy.ndarray): Gerçek etiketler.
        y_pred (numpy.ndarray): Modelin tahmin ettiği etiketler.

    Prints:
        Accuracy, precision, recall, f1-score.
    """
    accuracy = accuracy_score_custom(y_test, y_pred)
    precision = precision_score_custom(y_test, y_pred)
    recall = recall_score_custom(y_test, y_pred)
    f1 = f1_score_custom(y_test, y_pred)

    print(f"Doğruluk: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")

# Modelin performansını değerlendir
evaluate_model(y_test, y_pred)

# Karışıklık matrisini çiz
def plot_confusion_matrix(y_test, y_pred, title="Karışıklık Matrisi", save_path="results/confusion_matrix.png"):
    """
    Karışıklık matrisini çizer ve sonuçları görselleştirir.

    Args:
        y_test (numpy.ndarray): Gerçek etiketler.
        y_pred (numpy.ndarray): Modelin tahmin ettiği etiketler.
        title (str): Grafiğin başlığı.
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    cm = confusion_matrix_custom(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.xticks(np.arange(2), ['Negatif', 'Pozitif'])
    plt.yticks(np.arange(2), ['Negatif', 'Pozitif'])
    plt.grid(False)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.savefig(save_path)
    plt.show()

# Karışıklık matrisini çizdir
plot_confusion_matrix(y_test, y_pred)

# Karar sınırını çiz
def plot_decision_boundary(X, y, model, title, save_path="results/test_decision_boundary.png"):
    """
    Modelin karar sınırlarını çizdirir.

    Args:
        X (numpy.ndarray): Özellik matrisi.
        y (numpy.ndarray): Gerçek etiketler.
        model (keras.Model): Eğitilmiş model.
        title (str): Grafiğin başlığı.
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()

    plt.savefig(save_path)
    plt.show()

# Karar sınırını çizdir
plot_decision_boundary(X_test, y_test, model, title="Test Verisi ile Karar Sınırı")
