import os
import matplotlib.pyplot as plt

def add_features(df):
    df["CLOAD_sq"] = df["CLOAD (pF)"] ** 2
    return df

def plot_actual_vs_pred(y_true, y_pred, title, save_path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)