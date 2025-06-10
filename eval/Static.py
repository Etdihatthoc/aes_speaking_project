import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr

# --- Hàm tính các metrics giống trước ---
def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))

def root_mean_square_error(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def kappa(y_true, y_pred, weights='quadratic'):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)
    assert len(y_true) == len(y_pred)
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    y_true = y_true - min_rating
    y_pred = y_pred - min_rating
    num_ratings = max_rating - min_rating + 1
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
    num_items = float(len(y_true))
    weights_mat = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            diff = abs(i - j)
            if weights == 'linear':
                weights_mat[i, j] = diff
            elif weights == 'quadratic':
                weights_mat[i, j] = diff ** 2
            else:
                weights_mat[i, j] = bool(diff)
    hist_true = np.bincount(y_true, minlength=num_ratings) / num_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings) / num_items
    expected = np.outer(hist_true, hist_pred)
    conf_mat = conf_mat / num_items
    k = 1.0
    if np.count_nonzero(weights_mat):
        k -= np.sum(weights_mat * conf_mat) / np.sum(weights_mat * expected)
    return k

def pearson(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

def spearman(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

# --- Đường dẫn đến file CSV (thay thành file thật của bạn) ---
csv_path = '/mnt/disk1/SonDinh/SonDinh/aes_speaking_project/eval/results_multiscale_base_trainset.csv'
df = pd.read_csv(csv_path)

# Trích cột ground truth và predictions
y_true = df['vocabulary']
y_pred = df['predicted_vocabulary']

# Tính metrics và in ra
mae_value = mean_absolute_error(y_true, y_pred)
rmse_value = root_mean_square_error(y_true, y_pred)
qwk_value = kappa(y_true, y_pred, weights='quadratic')
pearson_value = pearson(y_true, y_pred)
spearman_value = spearman(y_true, y_pred)

print("Metrics:")
print(f"MAE: {mae_value:.4f}")
print(f"RMSE: {rmse_value:.4f}")
print(f"QWK (Quadratic Weighted Kappa): {qwk_value:.4f}")
print(f"Pearson correlation: {pearson_value:.4f}")
print(f"Spearman correlation: {spearman_value:.4f}")

# --- Tạo confusion matrix cho nhãn 0.0, 0.5, 1.0, ..., 10.0 ---

# 1. Làm tròn về bước 0.5
y_true_round = np.round(np.array(y_true) * 2) / 2
y_pred_round = np.round(np.array(y_pred) * 2) / 2

# 2. Chuyển sang index từ 0..20 bằng cách nhân với 2 rồi ép int
y_true_idx = (y_true_round * 2).astype(int)
y_pred_idx = (y_pred_round * 2).astype(int)

# 3. labels_idx là [0, 1, 2, ..., 20] tương ứng [0.0, 0.5, 1.0, ..., 10.0]
labels_idx = list(range(0, 21))

# 4. Tính confusion matrix trên các index
conf_mat_idx = confusion_matrix(y_true_idx, y_pred_idx, labels=labels_idx)

# 5. Tạo DataFrame để hiển thị, gắn nhãn lại về dạng float (bội 0.5)
labels_float = [i * 0.5 for i in labels_idx]
conf_df = pd.DataFrame(conf_mat_idx, index=labels_float, columns=labels_float)

print("\nConfusion Matrix (labels 0.0 to 10.0 step 0.5):")
print(conf_df)
