import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from joblib import dump, load  # For saving and loading the model

# Feature extraction functions
def extract_psd(signal, fs=512):
    f, psd = welch(signal, fs=fs, nperseg=256)
    return f, psd

def compute_mean_frequency(psd, freqs):
    return np.sum(freqs * psd) / np.sum(psd)

def compute_spectral_moments(psd, freqs):
    sm1 = compute_mean_frequency(psd, freqs)
    sm2 = np.sum(((freqs - sm1) ** 2) * psd) / np.sum(psd)
    sm3 = np.sum(((freqs - sm1) ** 3) * psd) / np.sum(psd)
    return sm1, sm2, sm3

def compute_power_spectrum_ratio(psd):
    peak_power = np.max(psd)
    total_power = np.sum(psd)
    return peak_power / total_power

def extract_features_from_channel(signal, fs=512):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128, nfft=512)
    spectrogram_flat = Sxx.flatten()
    f_psd, psd = extract_psd(signal, fs)
    mnf = compute_mean_frequency(psd, f_psd)
    sm1, sm2, sm3 = compute_spectral_moments(psd, f_psd)
    psr = compute_power_spectrum_ratio(psd)
    return np.hstack([mnf, sm1, sm2, sm3, psr, spectrogram_flat])

def extract_features_multichannel(data, fs=512):
    return np.hstack([extract_features_from_channel(data[:, ch], fs) for ch in range(data.shape[1])])

def load_data_from_files(data_dir, num_channels=8):
    X, y = [], []
    for label, folder in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        movement_data = np.load(file_path)
                        print(f"Loaded {filename}, shape: {movement_data.shape}")

                        # Ensure data has 8 channels
                        if len(movement_data.shape) == 3 and movement_data.shape[2] == num_channels:
                            # Expected shape: (samples, time, channels)
                            for sample in movement_data:
                                X.append(extract_features_multichannel(sample))
                                y.append(label)
                        elif len(movement_data.shape) == 2:
                            # Add channel dimension and replicate up to 8 channels
                            movement_data = np.expand_dims(movement_data, axis=-1)
                            if movement_data.shape[1] == num_channels:
                                for sample in movement_data:
                                    X.append(extract_features_multichannel(sample))
                                    y.append(label)
                            else:
                                print(f"Skipping {filename}: insufficient channels {movement_data.shape}")
                        else:
                            print(f"Skipping {filename}: unexpected shape {movement_data.shape}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
    
    if not X:
        raise ValueError("No valid data loaded. Check the directory and files.")
    
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()

def plot_cv_results(cv_scores):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(cv_scores[metric], label=metric.capitalize())
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross-Validation Metrics')
    plt.legend()
    plt.grid()
    plt.show()

# Set data directory
data_dir = r"C:\Users\simin\OneDrive\Desktop\Master an 1\TB\lab\sEMG spectrogram code\data"

# Load data with 8 channels enforced
features, labels = load_data_from_files(data_dir, num_channels=8)

# Check if data is valid
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Normalize features
if features.size > 0:
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
else:
    raise ValueError("Feature extraction failed. No features to normalize.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# Train SVM with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
print("Training SVM with GridSearch...")
svm.fit(X_train, y_train)

# Save the model
model_path = "svm_model.joblib"
dump(svm, model_path)
print(f"Model saved to {model_path}")

# Evaluate
print("Evaluating the model...")
y_pred = svm.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
class_names = [f"Movement {i+1}" for i in range(len(set(labels)))]
plot_confusion_matrix(y_test, y_pred, class_names)

# Perform k-fold cross-validation
print("Performing stratified k-fold cross-validation...")
skf = StratifiedKFold(n_splits=5)
cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for train_idx, val_idx in skf.split(features, labels):
    X_train_cv, X_val_cv = features[train_idx], features[val_idx]
    y_train_cv, y_val_cv = labels[train_idx], labels[val_idx]
    svm.fit(X_train_cv, y_train_cv)
    y_val_pred = svm.predict(X_val_cv)
    
    acc = svm.score(X_val_cv, y_val_cv)
    prec = precision_score(y_val_cv, y_val_pred, average='weighted')
    rec = recall_score(y_val_cv, y_val_pred, average='weighted')
    f1 = f1_score(y_val_cv, y_val_pred, average='weighted')
    
    cv_scores['accuracy'].append(acc)
    cv_scores['precision'].append(prec)
    cv_scores['recall'].append(rec)
    cv_scores['f1'].append(f1)
    print(f"Fold Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

print(f"Mean CV Metrics - Accuracy: {np.mean(cv_scores['accuracy']):.4f}, "
      f"Precision: {np.mean(cv_scores['precision']):.4f}, "
      f"Recall: {np.mean(cv_scores['recall']):.4f}, "
      f"F1: {np.mean(cv_scores['f1']):.4f}")

# Plot cross-validation metrics
plot_cv_results(cv_scores)
