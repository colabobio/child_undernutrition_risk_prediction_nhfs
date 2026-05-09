import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Dropout

# =============================
# Base directory (fixed paths)
# =============================
base_path = "C:/Users/justi/OneDrive/Desktop/feature-selection/min_node_size_5"

results = []

# =============================
# Loop through imp_1 → imp_15
# =============================
for i in range(1, 16):

    print(f"\n========== Running imp_{i} ==========\n")

    model_path = os.path.join(base_path, f"imp_{i}")

    # -------------------------
    # Load data
    # -------------------------
    X_train = pd.read_csv(os.path.join(model_path, "train_predictors.csv"))
    y_train = pd.read_csv(os.path.join(model_path, "train_response.csv")).values.ravel()

    X_test = pd.read_csv(os.path.join(model_path, "test_predictors.csv"))
    y_test = pd.read_csv(os.path.join(model_path, "test_response.csv")).values.ravel()

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nfeat = X_train.shape[1]

    # -------------------------
    # Model
    # -------------------------
    model = Sequential()
    model.add(Input(shape=(nfeat,)))

    model.add(Dense(nfeat, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nfeat // 2, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nfeat // 4, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # Train
    # -------------------------
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # -------------------------
    # Evaluate
    # -------------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Store results
    results.append({
        "dataset": f"imp_{i}",
        "accuracy": accuracy,
        "roc_auc": roc_auc
    })

    # -------------------------
    # ROC Curve
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'imp_{i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - imp_{i}')
    plt.legend()
    plt.show()

# =============================
# Final summary
# =============================
results_df = pd.DataFrame(results)

print("\n===== Overall Results =====")
print(results_df)

print("\nMean Accuracy:", results_df["accuracy"].mean())
print("Mean ROC-AUC:", results_df["roc_auc"].mean())