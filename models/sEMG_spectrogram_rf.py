import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, iirnotch, spectrogram
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------------------------
# 1) Funcții utile
# ----------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError("Critical frequencies must be between 0 and the Nyquist frequency.")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def notch_filter(data, freq, fs, quality=30):
    nyquist = 0.5 * fs
    freq = freq / nyquist
    b, a = iirnotch(freq, quality)
    return lfilter(b, a, data)

def calculate_psd(Sxx):
    return np.sum(Sxx, axis=1)

def calculate_entropy(Sxx):
    psd = calculate_psd(Sxx)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

def extract_features_from_spectrogram(f, t, Sxx):
    psd = calculate_psd(Sxx)
    ttp = np.sum(Sxx)  
    mnf = np.sum(f * psd) / np.sum(psd)
    sm1 = np.sum(f * psd) / np.sum(psd)
    sm2 = np.sum((f - sm1) ** 2 * psd) / np.sum(psd)
    sm3 = np.sum((f - sm1) ** 3 * psd) / np.sum(psd)
    psr = np.max(psd) / np.sum(psd)
    cf = np.argmax(psd) * (f[-1] - f[0]) / len(f)
    spectral_entropy = calculate_entropy(Sxx)

    return {
        "psd_sum": np.sum(psd),
        "sm1": sm1,
        "sm2": sm2,
        "sm3": sm3,
        "psr": psr,
        "spectral_entropy": spectral_entropy,
        "ttp": ttp,
    }

def extract_features():
    hand_dir = os.path.join(os.getcwd(), 'Hand')
    if not os.path.exists(hand_dir):
        raise FileNotFoundError(f"The directory {hand_dir} does not exist.")
    os.chdir(hand_dir)
    classes = os.listdir('.')
    features = []

    for id in range(len(classes)):
        class_path = os.path.join('.', classes[id])
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(classes[id]):
            if not file.endswith('.npy'):
                continue
            print(f"Procesăm fișierul: {file}")
            data = np.load(os.path.join(hand_dir, classes[id], file), allow_pickle=True)
            fs = 1024

            for i in range(data.shape[0]):
                signal = data[i, :]

                window_size = 5120
                overlap = int(window_size * 0.6)

                for start in range(0, len(signal) - window_size + 1, window_size - overlap):
                    window = signal[start:start + window_size]

                    filtered_signal = apply_bandpass_filter(window, 10, 500, fs)
                    filtered_signal = notch_filter(filtered_signal, 50, fs)

                    f, t, Sxx = spectrogram(filtered_signal, fs, nperseg=256, noverlap=128)

                    feature_set = extract_features_from_spectrogram(f, t, Sxx)
                    feature_set["class"] = classes[id]
                    feature_set["channel"] = i + 1
                    features.append(feature_set)

    feature_df = pd.DataFrame(features)
    os.chdir('..')
    return feature_df

# ----------------------------------------------------------------------
# 2) Main script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # A) Extragem caracteristicile
    features_df = extract_features()
    features_df.to_csv('../features_rf.csv', index=False)
    print("Caracteristicile au fost salvate în features_rf.csv")

    # B) Încărcăm datele
    features_df = pd.read_csv('../features_rf.csv')
    X = features_df.drop(columns=["class", "channel"])
    y = features_df["class"]

    # C) Normalizare
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # D) Oversampling cu SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # E) Împărțire: 80% train, 10% val, 10% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    val_size = 0.1 / 0.9  # ~0.1111
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=42, stratify=y_trainval
    )

    print("\nDimensiuni seturi:")
    print(f"Train = {len(y_train)}")
    print(f"Val   = {len(y_val)}")
    print(f"Test  = {len(y_test)}")

    # F) Model Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # G) Cross-validare (opțional) pe setul de antrenament
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print("\nAcuratețea medie pe cross-validare (Train): "
          f"{cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # H) Antrenare pe tot setul de antrenament
    rf_model.fit(X_train, y_train)

    # I) Evaluare pe setul de validare
    y_val_pred = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nAcuratețea pe setul de validare: {val_acc:.2f}")

    val_report = classification_report(y_val, y_val_pred)
    print("\nClassification Report (Validation):")
    print(val_report)

    # Salvăm classification report (Val) într-un fișier
    with open("rf_validation_classification_report.txt", "w") as f_val:
        f_val.write(val_report)

    # J) Evaluare pe setul de test
    y_test_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\nAcuratețea pe setul de test: {test_acc:.2f}")

    test_report = classification_report(y_test, y_test_pred)
    print("\nClassification Report (Test):")
    print(test_report)

    # Salvăm classification report (Test) într-un fișier
    with open("rf_test_classification_report.txt", "w") as f_test:
        f_test.write(test_report)

    # K) Matricea de confuzie pe Test
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=rf_model.classes_,
        yticklabels=rf_model.classes_
    )
    plt.title("Matricea de confuzie - Random Forest (Test)")
    plt.xlabel("Predicții")
    plt.ylabel("Adevăr")
    plt.show()
