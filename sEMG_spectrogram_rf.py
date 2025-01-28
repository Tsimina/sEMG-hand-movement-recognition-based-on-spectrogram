import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.svm import SVC  # Nu mai folosim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, iirnotch, spectrogram
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Funcții utile
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
    """Returnează PSD-ul prin sumarea spectrului pe axa 1."""
    return np.sum(Sxx, axis=1)

def calculate_entropy(Sxx):
    """Calculează entropia pe PSD normalizat."""
    psd = calculate_psd(Sxx)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

def extract_features_from_spectrogram(f, t, Sxx):
    """Extrage câteva caracteristici din spectrogramă: PSD, MNF, moment spectral, entropie etc."""
    psd = calculate_psd(Sxx)
    ttp = np.sum(Sxx)  # total time power
    mnf = np.sum(f * psd) / np.sum(psd)
    sm1 = np.sum(f * psd) / np.sum(psd)
    sm2 = np.sum((f - sm1) ** 2 * psd) / np.sum(psd)
    sm3 = np.sum((f - sm1) ** 3 * psd) / np.sum(psd)
    psr = np.max(psd) / np.sum(psd)
    cf = np.argmax(psd) * (f[-1] - f[0]) / len(f)
    spectral_entropy = calculate_entropy(Sxx)

    return {
        "psd_sum": np.sum(psd),
        "mnf": mnf,
        "sm1": sm1,
        "sm2": sm2,
        "sm3": sm3,
        "psr": psr,
        "central_frequency": cf,
        "spectral_entropy": spectral_entropy,
        "ttp": ttp,
    }

def extract_features():
    """Parcurge fișierele din folderul 'Hand', extrage ferestre din semnal, calculează spectrograma și caracteristicile."""
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
        for file in os.listdir(class_path):
            if not file.endswith('.npy'):
                continue
            print(file)
            data = np.load(os.path.join(hand_dir, classes[id], file), allow_pickle=True)
            fs = 1024

            for i in range(data.shape[0]):
                signal = data[i, :]

                # Împărțirea semnalului în ferestre
                window_size = 5120  # exemplu (5 secunde la 1024 Hz)
                overlap = int(window_size * 0.6)

                for start in range(0, len(signal) - window_size + 1, window_size - overlap):
                    window = signal[start:start + window_size]

                    # Filtrare semnal
                    filtered_signal = apply_bandpass_filter(window, 10, 500, fs)
                    filtered_signal = notch_filter(filtered_signal, 50, fs)

                    # Calculul spectrogramelor
                    f, t, Sxx = spectrogram(filtered_signal, fs, nperseg=256, noverlap=128)

                    # Extracția caracteristicilor
                    feature_set = extract_features_from_spectrogram(f, t, Sxx)
                    feature_set["class"] = classes[id]
                    feature_set["channel"] = i + 1
                    features.append(feature_set)

    feature_df = pd.DataFrame(features)
    os.chdir('..')
    return feature_df


features_df = extract_features()
features_df.to_csv('../features_rf.csv', index=False)
print("Caracteristicile au fost salvate în features_rf.csv")


features_df = pd.read_csv('../features_rf.csv')
X = features_df.drop(columns=["class", "channel"])
y = features_df["class"]

# Normalizare
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Oversampling cu SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=100,  # Poți ajusta
    random_state=42
)

cross_val_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("\nAcuratețea medie pe cross-validare (train): {:.2f} ± {:.2f}".format(
    cross_val_scores.mean(), cross_val_scores.std())
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("\nAcuratețea pe setul de test:", accuracy_score(y_test, y_pred))

report_str = classification_report(y_test, y_pred)
print("\nRaport de clasificare:")
print(report_str)

with open("results_rf.txt", "w") as f:
    f.write(report_str)

# Matricea de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)
plt.title("Matricea de confuzie - Random Forest")
plt.xlabel("Predicții")
plt.ylabel("Adevăr")
plt.show()
