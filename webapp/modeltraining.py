import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import io
import hashlib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# ===== Add near your existing imports =====
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau  # you already import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense, Concatenate
from tensorflow.keras.models import Model


# --- Helpers ---
def z_score_per_sample(X):
    """Per-sample z-score over time axis. X: (N, T, C)."""
    X = np.asarray(X, dtype=np.float32)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s

@keras.utils.register_keras_serializable()
class SinusoidalPositionalEncoding(layers.Layer):
    """Non-trainable Transformer PE (sin/cos)."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def call(self, x):
        # x: (B, T, D)
        T = tf.shape(x)[1]
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)   # (1,D)
        pos = tf.cast(tf.range(T)[:, tf.newaxis], tf.float32)            # (T,1)
        angles = pos / tf.pow(10000.0, (2.0 * tf.floor(i/2.0)) / float(self.d_model))
        pe = tf.where(tf.cast(i % 2, tf.bool), tf.cos(angles), tf.sin(angles))  # (T,D)
        return x + pe[tf.newaxis, :, :]                                  # (B,T,D)

def transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=128, dropout=0.1, l2w=0.0):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    y = layers.Conv1D(ff_dim, 1, activation="relu",
                      kernel_regularizer=regularizers.l2(l2w) if l2w > 0 else None)(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Conv1D(x.shape[-1], 1,
                      kernel_regularizer=regularizers.l2(l2w) if l2w > 0 else None)(y)
    y = layers.LayerNormalization(epsilon=1e-6)(y + x)
    return y

def build_ts_transformer_binary(seq_len, n_channels,
                                d_model=64, head_size=32, num_heads=2, ff_dim=128,
                                num_transformer_blocks=2, dropout=0.1,
                                mlp_units=(64,), mlp_dropout=0.1,
                                learning_rate=1e-3):
    inputs = layers.Input(shape=(seq_len, n_channels))
    x = layers.Conv1D(d_model, 1, padding="same")(inputs)
    x = SinusoidalPositionalEncoding(d_model=d_model)(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    x = layers.GlobalAveragePooling1D()(x)
    for u in (mlp_units if isinstance(mlp_units, (list, tuple)) else [mlp_units]):
        x = layers.Dense(u, activation="relu")(x); x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_ts_transformer_multiclass(seq_len, n_channels, n_classes,
                                    d_model=64, head_size=32, num_heads=2, ff_dim=128,
                                    num_transformer_blocks=2, dropout=0.2, l2w=1e-4,
                                    mlp_units=(128,), mlp_dropout=0.3,
                                    learning_rate=1e-3):
    inputs = layers.Input(shape=(seq_len, n_channels))
    x = layers.Conv1D(d_model, 1, padding="same",
                      kernel_regularizer=regularizers.l2(l2w))(inputs)
    x = SinusoidalPositionalEncoding(d_model=d_model)(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size=head_size, num_heads=num_heads,
                                ff_dim=ff_dim, dropout=dropout, l2w=l2w)
    # dual pooling
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x),
                              layers.GlobalMaxPooling1D()(x)])
    for u in (mlp_units if isinstance(mlp_units, (list, tuple)) else [mlp_units]):
        x = layers.Dense(u, activation="relu")(x); x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_inception_model(input_shape, num_classes=1):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    def inception_module(x, filters):
        conv1 = tf.keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
        conv3 = tf.keras.layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        conv5 = tf.keras.layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
        pool = tf.keras.layers.MaxPooling1D(3, strides=1, padding='same')(x)
        pool_conv = tf.keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)
        return tf.keras.layers.Concatenate()([conv1, conv3, conv5, pool_conv])

    x = inception_module(input_layer, 32)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = inception_module(x, 64)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = inception_module(x, 128)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Decide activation based on number of classes
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output_layer = tf.keras.layers.Dense(num_classes, activation=activation)(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model


def to_3d(X):
    """
    Ensure X is (n_samples, timesteps, 1).
    Accepts pandas DataFrame (first column holds Series) or numpy.ndarray.
    """
    if isinstance(X, pd.DataFrame):
        # sktime style: first column holds pd.Series
        series_list = [np.asarray(s) for s in X.iloc[:, 0].tolist()]
        arr = np.stack(series_list, axis=0)             # (n_samples, timesteps)
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            arr = X[:, None]                            # (n_samples, 1)
        elif X.ndim == 2:
            arr = X                                     # (n_samples, timesteps)
        elif X.ndim == 3:
            return X                                    # already (n, t, c)
        else:
            raise ValueError(f"Unexpected ndarray ndim={X.ndim}")
    else:
        raise TypeError("X must be a DataFrame or NumPy array")
    return arr[..., np.newaxis]                         # (n_samples, timesteps, 1)

def _model_key(ds_id, lr, epochs, batch_size):
    # Deterministic key even if floats have different str formatting
    payload = json.dumps({
        "ds_id": int(ds_id),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch_size)
    }, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]  # short, stable id

def _save_artifacts(model, ds_id, lr, epochs, batch_size, metrics_dict):
    import os, io, json

    key = _model_key(ds_id, lr, epochs, batch_size)
    base_dir = os.path.join("saved_models", f"ds{ds_id}_{lr}_{epochs}_{batch_size}", key)
    os.makedirs(base_dir, exist_ok=True)

    # 1) Save the model
    model_path = os.path.join(base_dir, "model.keras")
    model.save(model_path)

    # 2) Save metrics (UTF-8 + keep non-ASCII)
    with open(os.path.join(base_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # 3) Save metadata (UTF-8)
    meta = {
        "ds_id": int(ds_id),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch_size)
    }
    with open(os.path.join(base_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 4) Save summary text (UTF-8)
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    with open(os.path.join(base_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(s.getvalue())

    return {"key": key, "dir": base_dir, "model_path": model_path}

def train_model(X_train, y_train, X_test, y_test, epochs, lr, batch_size, ds_id, model='cnn', threshold=0.5):
    """
    Trains a 1D CNN model and returns the test accuracy.
    """

    # Define model architecture based on dataset ID
    if ds_id == 1:  # ItalyPowerDemand
        # Pre-processing the dataset before training the model

        # Convert to 3D NumPy arrays

        X_train_np = to_3d(X_train)
        X_test_np = to_3d(X_test)

        #  Z-score standardization (per time series)
        def z_score_standardize(X):
            X_std = np.zeros_like(X)
            for i in range(X.shape[0]):
                ts = X[i, :, 0]
                mean = np.mean(ts)
                std = np.std(ts)
                if std == 0:
                    std = 1  # avoid division by zero
                X_std[i, :, 0] = (ts - mean) / std
            return X_std

        X_train_np = z_score_standardize(X_train_np)
        X_test_np = z_score_standardize(X_test_np)

        # Encode class labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        y_train_cat = to_categorical(y_train_enc)
        y_test_cat = to_categorical(y_test_enc)

        if model == 'cnn':
            y_train_cat = to_categorical(y_train_enc)
            y_test_cat = to_categorical(y_test_enc)
            net = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(X_train_np.shape[1], 1)),
                MaxPooling1D(2),
                Dropout(0.3),
                Conv1D(128, 3, activation='relu'),
                MaxPooling1D(2),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.2),
                Dense(y_train_cat.shape[1], activation='softmax')
            ])
            net.compile(optimizer=Adam(learning_rate=lr),
                        loss='categorical_crossentropy', metrics=['accuracy'])
        elif model== 'transformer':
            y_train_bin = y_train_enc.astype('float32').reshape(-1, 1)
            net = build_ts_transformer_binary(
                seq_len=X_train_np.shape[1], n_channels=1,
                learning_rate=lr
            )
        elif model in ('inception', 'inceptiontime'):
            y_train_bin = y_train_enc.astype('float32').reshape(-1, 1)
            net = build_inception_model(X_train_np.shape[1:],num_classes=1)
            net.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)

        if model == 'cnn':
            net.fit(X_train_np, y_train_cat, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, callbacks=[es, rlr], verbose=0)
            y_prob = net.predict(X_test_np, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
        else:
            net.fit(X_train_np, y_train_bin, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, callbacks=[es, rlr], verbose=0)
            y_prob = net.predict(X_test_np, verbose=0).ravel()
            y_pred = (y_prob > threshold).astype(int)

        save_path = os.path.join("saved_models", f"{model}_ds{ds_id}_{lr}_{epochs}_{batch_size}.keras")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        net.save(save_path)

        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test_enc, y_pred, zero_division=0)


        return {
            "report": report,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        }



    elif ds_id == 2:  # CinCECGTorso
        X_train_np = to_3d(X_train)
        X_test_np = to_3d(X_test)

        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        #y_train_cat = to_categorical(y_train_enc)
        #y_test_cat = to_categorical(y_test_enc)

        n_classes = int(np.unique(y_train_enc).size)

        if model == 'cnn':
            y_train_cat = to_categorical(y_train_enc, num_classes=n_classes)
            net = Sequential([
                Conv1D(64, 5, activation='relu', input_shape=(X_train_np.shape[1], 1)),
                MaxPooling1D(2),
                Dropout(0.3),
                Conv1D(128, 3, activation='relu'),
                MaxPooling1D(2),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.2),
                Dense(n_classes, activation='softmax')
            ])
            net.compile(optimizer=Adam(learning_rate=lr),
                        loss='categorical_crossentropy', metrics=['accuracy'])
            Xtr_in, Ytr_in = X_train_np, y_train_cat
        elif model == 'transformer':
            # Transformer in your notebook uses per-sample z-score + L2 + dual pooling
            X_train_np = z_score_per_sample(X_train_np)
            X_test_np = z_score_per_sample(X_test_np)
            y_train_cat = to_categorical(y_train_enc, num_classes=n_classes)
            net = build_ts_transformer_multiclass(
                seq_len=X_train_np.shape[1], n_channels=1, n_classes=n_classes,
                learning_rate=lr
            )
            Xtr_in, Ytr_in = X_train_np, y_train_cat
        elif model in ('inception', 'inceptiontime'):
            X_train_np = z_score_per_sample(X_train_np)
            X_test_np = z_score_per_sample(X_test_np)
            y_train_cat = to_categorical(y_train_enc, num_classes=n_classes)
            net = build_inception_model(X_train_np.shape[1:],num_classes=4)
            net.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            Xtr_in, Ytr_in = X_train_np, y_train_cat

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)

        net.fit(Xtr_in, Ytr_in, epochs=epochs, batch_size=batch_size,
                validation_split=0.2, callbacks=[es, rlr], verbose=0)


        y_prob = net.predict(X_test_np, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        save_path = os.path.join("saved_models", f"{model}_ds{ds_id}_{lr}_{epochs}_{batch_size}.keras")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        net.save(save_path)

        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test_enc, y_pred, zero_division=0)
        return {
            "report": report,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        }



    elif ds_id == 3:  # ECG200
        # 2D -> 3D
        X_train_2d = np.stack(
            X_train if isinstance(X_train, (list, np.ndarray)) else X_train.iloc[:, 0].apply(np.asarray).values)
        X_test_2d = np.stack(
            X_test if isinstance(X_test, (list, np.ndarray)) else X_test.iloc[:, 0].apply(np.asarray).values)
        X_train_np = X_train_2d.reshape((X_train_2d.shape[0], X_train_2d.shape[1], 1))
        X_test_np = X_test_2d.reshape((X_test_2d.shape[0], X_test_2d.shape[1], 1))

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # class weights (used in your notebook)
        cw = compute_class_weight(class_weight='balanced',
                                  classes=np.unique(y_train_enc),
                                  y=y_train_enc)
        class_weights = dict(zip(np.unique(y_train_enc), cw))

        if model == 'cnn':
            net = keras.Sequential([
                keras.layers.InputLayer(input_shape=(X_train_np.shape[1], 1)),
                keras.layers.Conv1D(32, 5, activation='relu'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            net.compile(optimizer=Adam(learning_rate=lr),
                        loss='binary_crossentropy', metrics=['accuracy'])
        elif model=='transformer':
            # Transformer + per-sample z-score
            X_train_np = z_score_per_sample(X_train_np)
            X_test_np = z_score_per_sample(X_test_np)
            net = build_ts_transformer_binary(
                seq_len=X_train_np.shape[1], n_channels=1, learning_rate=lr
            )
        elif model in ('inception', 'inceptiontime'):
            X_train_np = z_score_per_sample(X_train_np)
            X_test_np = z_score_per_sample(X_test_np)
            net = build_inception_model(X_train_np.shape[1:],num_classes=1)
            net.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)

        net.fit(X_train_np, y_train_enc if model == 'cnn' else y_train_enc.astype('float32').reshape(-1, 1),
                epochs=epochs, batch_size=batch_size, verbose=0,
                validation_split=0.2, callbacks=[es, rlr],
                class_weight=class_weights if model == 'cnn' else class_weights)

        y_prob = net.predict(X_test_np, verbose=0).ravel()
        y_pred = (y_prob > threshold).astype(int)

        save_path = os.path.join("saved_models", f"{model}_ds{ds_id}_{lr}_{epochs}_{batch_size}.keras")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        net.save(save_path)

        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test_enc, y_pred, zero_division=0)

        return {
            "report": report,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        }




