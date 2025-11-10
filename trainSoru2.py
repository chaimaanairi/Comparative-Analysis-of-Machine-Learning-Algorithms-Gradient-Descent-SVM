import os
import numpy as np
import matplotlib.pyplot as plt
from custom_metrics import confusion_matrix_custom, accuracy_score_custom, precision_score_custom, recall_score_custom, f1_score_custom
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from dataset import load_data

# Klasörleri kontrol et ve oluştur
os.makedirs("results", exist_ok=True)

def plot_loss(history, title, filename):
    """
    Eğitim ve doğrulama kayıplarını grafikte gösterir ve kaydeder.

    Parametreler:
    history -- Modelin eğitim sırasında kaydedilen geçmişi (loss, accuracy vb.)
    title -- Grafiğin başlığı
    filename -- Grafiğin kaydedileceği dosya adı
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Eğitim Loss')
    plt.plot(history.history['val_loss'], label='Doğrulama Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"results/{filename}")
    plt.show()

def plot_decision_boundary(model, X, y, title, filename):
    """
    Modelin karar sınırlarını görselleştirir ve kaydeder.

    Parametreler:
    model -- Eğitilmiş yapay zeka modeli
    X -- Özellikler (girdi verisi)
    y -- Etiketler (girdi verisinin sınıfları)
    title -- Grafiğin başlığı
    filename -- Grafiğin kaydedileceği dosya adı
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"results/{filename}")
    plt.show()

def build_model(hidden_layers):
    """
    Yapay Nöron Ağı modelini oluşturur.

    Parametre:
    hidden_layers -- Gizli katman sayısı

    Dönen Değer:
    model -- Keras Sequential modeli
    """
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='sigmoid'))

    for _ in range(hidden_layers - 1):
        model.add(Dense(8, activation='sigmoid'))

    model.add(Dense(1, activation='sigmoid'))
    return model

def train_and_evaluate(hidden_layers, optimizer, batch_size=None):
    """
    Modeli eğitir ve sonuçları değerlendirir.

    Parametreler:
    hidden_layers -- Gizli katman sayısı
    optimizer -- Modelin optimizasyonunda kullanılacak optimizasyon algoritması
    batch_size -- Minibatch boyutu (None için tam batch kullanılacaktır)
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Model oluşturma
    model = build_model(hidden_layers)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Eğitim
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        verbose=0
    )

    # Loss grafiği çizimi
    optimizer_name = optimizer.__class__.__name__
    batch_info = f"batch_size={batch_size}" if batch_size else "full-batch"
    title = f"{hidden_layers} Katmanlı Model ({optimizer_name}, {batch_info})"
    filename = f"loss_{hidden_layers}_layers_{optimizer_name}_{batch_info}.png"
    plot_loss(history, title, filename)

    # Karar sınırlarının görselleştirilmesi
    decision_boundary_file = f"decision_boundary_{hidden_layers}_layers_{optimizer_name}_{batch_info}.png"
    plot_decision_boundary(model, X_train, y_train, title, decision_boundary_file)

    # Değerlendirme Metrikleri
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    cm = confusion_matrix_custom(y_test, y_pred)
    acc = accuracy_score_custom(y_test, y_pred)
    prec = precision_score_custom(y_test, y_pred)
    rec = recall_score_custom(y_test, y_pred)
    f1 = f1_score_custom(y_test, y_pred)

    print(f"{title} Sonuçları:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}\n")

    return model

if __name__ == "__main__":
    # Eğitim başlatma ve farklı optimizer'lar ile model eğitimi
    # SGD için eğitim
    print("Stochastic Gradient Descent ile Eğitim Başlıyor...")
    for layers in [1, 2, 3]:
        train_and_evaluate(hidden_layers=layers, optimizer=SGD(learning_rate=0.01))

    # Batch Gradient Descent için eğitim
    print("Batch Gradient Descent ile Eğitim Başlıyor...")
    for layers in [1, 2, 3]:
        train_and_evaluate(hidden_layers=layers, optimizer=SGD(learning_rate=0.01), batch_size=400)

    # Mini Batch Gradient Descent için eğitim
    print("Mini Batch Gradient Descent ile Eğitim Başlıyor...")
    for layers in [1, 2, 3]:
        train_and_evaluate(hidden_layers=layers, optimizer=SGD(learning_rate=0.01), batch_size=32)
