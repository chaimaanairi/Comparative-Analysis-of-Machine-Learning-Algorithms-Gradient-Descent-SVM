import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Klasörleri kontrol et ve oluştur
os.makedirs("dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)


def create_and_save_dataset():
    """
    make_moons fonksiyonu ile ikili sınıflı veri kümesi oluşturur ve eğitim, doğrulama, test verisi olarak böler.
    Ayrıca, bu verileri .npy formatında dosyalara kaydeder ve veri kümesini görselleştirir.

    Bu fonksiyon, sklearn'un `make_moons` fonksiyonunu kullanarak, ikili sınıf problemi için bir veri kümesi oluşturur.
    Veri kümesi oluşturulduktan sonra, eğitim, doğrulama ve test setlerine bölünür ve bunlar 'dataset' klasörüne kaydedilir.
    Ayrıca, oluşturulan veri kümesinin görselleştirmesi yapılır ve 'results' klasörüne kaydedilir.
    """
    # Veri kümesi oluşturma
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

    # Eğitim, doğrulama ve test verilerini ayırma
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Eğitim/Doğrulama/Test verilerini kaydet
    np.save("dataset/X_train.npy", X_train)
    np.save("dataset/y_train.npy", y_train)
    np.save("dataset/X_val.npy", X_val)
    np.save("dataset/y_val.npy", y_val)
    np.save("dataset/X_test.npy", X_test)
    np.save("dataset/y_test.npy", y_test)

    # Çizim - Veri kümesini görselleştir
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.title("İkili Sınıflı Veri Kümesi")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()

    # Grafiği kaydet
    plt.savefig("results/dataset_visualization.png")
    plt.show()

    # Eğitim, doğrulama ve test verilerini yazdır
    print(f"Eğitim Verisi: (%60) {X_train.shape[0]} örnek")
    print(f"Doğrulama Verisi: (%20) {X_val.shape[0]} örnek")
    print(f"Test Verisi: (%20) {X_test.shape[0]} örnek")

    print(f"Dataset başarıyla oluşturuldu ve 'dataset' ile 'results' dizinine kaydedildi.")


def load_data():
    """
    Veri kümesini 'dataset' klasöründen yükler ve döndürür.

    Bu fonksiyon, daha önce 'create_and_save_dataset' fonksiyonu ile kaydedilen veri kümesini
    yükler ve eğitim, doğrulama ve test setlerini döndüren bir fonksiyondur.

    Dönen Değerler:
    X_train -- Eğitim verisinin özellikleri
    y_train -- Eğitim verisinin etiketleri
    X_val -- Doğrulama verisinin özellikleri
    y_val -- Doğrulama verisinin etiketleri
    X_test -- Test verisinin özellikleri
    y_test -- Test verisinin etiketleri
    """
    X_train = np.load("dataset/X_train.npy")
    y_train = np.load("dataset/y_train.npy")
    X_val = np.load("dataset/X_val.npy")
    y_val = np.load("dataset/y_val.npy")
    X_test = np.load("dataset/X_test.npy")
    y_test = np.load("dataset/y_test.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    create_and_save_dataset()
