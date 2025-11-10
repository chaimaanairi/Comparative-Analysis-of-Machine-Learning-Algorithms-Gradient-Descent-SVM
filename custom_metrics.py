import numpy as np

# Function to calculate Confusion Matrix
def confusion_matrix_custom(y_true, y_pred):
    """
        Karışıklık matrisini hesaplar.
        Binary sınıflandırma için geçerlidir.

        Args:
            y_true (numpy.ndarray): Gerçek etiketler.
            y_pred (numpy.ndarray): Tahmin edilen etiketler.

        Returns:
            numpy.ndarray: Karışıklık matrisi, [[TP, FP], [FN, TN]] formatında.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    return np.array([[tp, fp], [fn, tn]])

# Function to calculate Accuracy
def accuracy_score_custom(y_true, y_pred):
    """
        Doğruluk skorunu hesaplar.

        Args:
            y_true (numpy.ndarray): Gerçek etiketler.
            y_pred (numpy.ndarray): Tahmin edilen etiketler.

        Returns:
            float: Doğruluk skoru.
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Function to calculate Precision
def precision_score_custom(y_true, y_pred):
    """
        Kesinlik (Precision) skorunu hesaplar.

        Args:
            y_true (numpy.ndarray): Gerçek etiketler.
            y_pred (numpy.ndarray): Tahmin edilen etiketler.

        Returns:
            float: Kesinlik skoru.
    """
    cm = confusion_matrix_custom(y_true, y_pred)
    tp = cm[0, 0]
    fp = cm[0, 1]
    return tp / (tp + fp) if (tp + fp) != 0 else 0

# Function to calculate Recall
def recall_score_custom(y_true, y_pred):
    """
        Duyarlılık (Recall) skorunu hesaplar.

        Args:
            y_true (numpy.ndarray): Gerçek etiketler.
            y_pred (numpy.ndarray): Tahmin edilen etiketler.

        Returns:
            float: Duyarlılık skoru.
    """
    cm = confusion_matrix_custom(y_true, y_pred)
    tp = cm[0, 0]
    fn = cm[1, 0]
    return tp / (tp + fn) if (tp + fn) != 0 else 0

# Function to calculate F1-Score
def f1_score_custom(y_true, y_pred):
    """
        F1-skorunu hesaplar.

        Args:
            y_true (numpy.ndarray): Gerçek etiketler.
            y_pred (numpy.ndarray): Tahmin edilen etiketler.

        Returns:
            float: F1-skoru.
    """
    precision = precision_score_custom(y_true, y_pred)
    recall = recall_score_custom(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
