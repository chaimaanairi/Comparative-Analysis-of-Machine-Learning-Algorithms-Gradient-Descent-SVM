## Açıklama
Makine Öğrenmesi (BLM5110) dersi kapsamında TensorFlow, Keras ve Scikit-learn kütüphaneleri kullanılarak geliştirilen sınıflandırma modelini içeren proje.
Bu projede, Destek Vektör Makinesi (SVM) ve Stokastik Gradient Descent (SGD) gibi farklı makine öğrenmesi algoritmaları kullanılarak çeşitli kernel fonksiyonları (lineer, polinomsal, Gaussian RBF) ile model eğitilmiştir ve değerlendirilmiştir.


## Gereksinimler
Bu projeyi çalıştırabilmek için gerekli Python kütüphaneleri requirements.txt dosyasında belirtilmiştir.
İlgili kütüphaneleri yüklemek için terminal veya komut satırına aşağıdaki komutu giriniz:
'pip install -r requirements.txt'


## Çalıştırma
2. Soru için Eğitim:
Modelin eğitilmesi için aşağıdaki komut çalıştırılmalıdır:
'python trainSoru2.py'
Bu komut, ilk modelin eğitilmesini ve ilgili parametrelerin Grid Search ile optimize edilmesini sağlar.

3. Soru için Eğitim:
Modelin eğitilmesi için aşağıdaki komut çalıştırılmalıdır:
'python trainSoru3.py'
Bu komut, Destek Vektör Makinesi (SVM) ile farklı kernel fonksiyonları (lineer, polinomsal, Gaussian RBF) kullanılarak Grid Search ve model optimizasyonu yapılır.


## Değerlendirme:
Eğitilen modelin performansını değerlendirmek için aşağıdaki komut çalıştırılmalıdır:
'python eval.py'
Bu komut, test verisi ile modelin doğruluk, precision, recall ve F1 skoru gibi metriklerini hesaplar.
Ayrıca karar sınırlarını ve karışıklık matrisini görselleştirir.


## Dosya Düzeni
Proje dosya yapısı aşağıdaki gibi olacaktır:
proje_klasoru/
│
├── dataset.py              # Veri kümesini yüklemek için kullanılan dosya
├── trainSoru2.py           # 2. soru için model eğitimi dosyası (SGD ile optimizasyon)
├── trainSoru3.py           # 3. soru için SVM model eğitimi dosyası (Farklı kernel fonksiyonları ile Grid Search)
├── eval.py                 # Eğitilen modelin değerlendirilmesini gerçekleştiren dosya
├── custom_metrics.py       # confusion_matrix_custom, accuracy_score_custom, precision_score_custom, recall_score_custom, f1_score_custom hesaplayan dosya
├── requirements.txt        # Gerekli Python kütüphanelerinin listelendiği dosya
│
├── dataset/                # Eğitim ve test verilerinin bulunduğu klasör
└── results/                # Eğitilmiş model ve değerlendirme sonuçlarının kaydedildiği klasör


## Ekstra Bilgiler

trainSoru2.py: Bu dosya, ikinci soru kapsamında Stokastik Gradient Descent (SGD) gibi optimizasyon teknikleriyle modelin eğitilmesini sağlar.

trainSoru3.py: Bu dosya, üçüncü soru kapsamında Destek Vektör Makineleri (SVM) ile farklı kernel fonksiyonları (lineer, polinomsal, Gaussian RBF) kullanarak modelin eğitilmesini ve Grid Search ile optimizasyon yapılmasını sağlar.

eval.py: Bu dosya, eğitilen modelin doğrulama ve test setlerinde performansını ölçer. Doğruluk, precision, recall, F1 skoru gibi metrikleri hesaplar ve karışıklık matrisi ile karar sınırlarını görselleştirir.

requirements.txt: Bu dosya, proje için gerekli olan Python kütüphanelerini listeler. Bağımlılıkları yüklemek için kullanılır.

custom_metrics.py: Bu dosya, confusion matrix, accuracy, precision, recall ve F1 skorlarını hesaplayan özel fonksiyonları içerir. Bu fonksiyonlar, scikit-learn gibi kütüphanelerin hazır fonksiyonlarını kullanmadan metriklerin hesaplanmasını sağlar.


