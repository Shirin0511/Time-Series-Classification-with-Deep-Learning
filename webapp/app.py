from visualizations import generate_visualizations
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from flask import Flask, render_template, send_file, request, jsonify
from modeltraining import train_model
import os, io, json, hashlib
from flask import request, jsonify
#from tensorflow.keras.models import load_model
from keras.saving import load_model as keras_load_model
from sklearn.preprocessing import LabelEncoder  # for re-encoding y
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from modeltraining import SinusoidalPositionalEncoding



app = Flask(__name__)
dataset_cache = {}


# This function will now be responsible for caching the datasets
def load_dataset_cached(dataset_id):
    """Loads a dataset from file and caches it."""
    if dataset_id in dataset_cache:
        print(f"Loading dataset {dataset_id} from cache.")
        return dataset_cache[dataset_id]

    X_train, y_train = None, None
    X_test, y_test = None, None

    if dataset_id == 1:  # Italy Dataset
        X_train, y_train = load_from_tsfile_to_dataframe('data/ItalyPowerDemand_TRAIN.ts')
        X_test, y_test = load_from_tsfile_to_dataframe('data/ItalyPowerDemand_TEST.ts')
    elif dataset_id == 2:  # Add your other datasets here
        X_train, y_train = load_from_tsfile_to_dataframe('data/CinCECGTorso_TRAIN.ts')
        X_test, y_test = load_from_tsfile_to_dataframe('data/CinCECGTorso_TEST.ts')
    elif dataset_id == 3:  # Add your other datasets here
        X_train, y_train = load_from_tsfile_to_dataframe('data/ECG200_TRAIN.ts')
        X_test, y_test = load_from_tsfile_to_dataframe('data/ECG200_TEST.ts')


    if X_train is not None:
        dataset_cache[dataset_id] = (X_train, y_train, X_test, y_test)
        print(f"Dataset {dataset_id} loaded and cached.")
        return dataset_cache[dataset_id]

    return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userguide')
def userguide():
    return render_template('userguide.html')




@app.route('/get_visualization/<int:ds_id>')
def get_visualization(ds_id):
    try:
        # Pass the dataset ID to your plotting function
        image_buffer = generate_visualizations(ds_id)
        # Serve the in-memory image buffer as a PNG file
        return send_file(image_buffer, mimetype='image/png', as_attachment=False)
    except ValueError as e:
        return str(e), 404





@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    try:
        data = request.get_json()
        ds_id = int(data.get('ds_id'))
        epochs = int(data.get('epochs'))
        lr = float(data.get('lr'))
        batch_size = int(data.get('batch_size'))
        model_choice = (data.get('model') or 'cnn')

        if not all([ds_id, epochs, lr, batch_size]):
            return jsonify({'error': 'Missing training parameters'}), 400

        # Load the dataset
        X_train_df, y_train, X_test_df, y_test = load_dataset_cached(ds_id)

        if X_train_df is None:
            return jsonify({'error': f'Dataset not found for ID {ds_id}'}), 404

        # Convert sktime DF -> numpy arrays
        X_train = np.stack(X_train_df.iloc[:, 0].apply(lambda x: x.values).values)
        X_test = np.stack(X_test_df.iloc[:, 0].apply(lambda x: x.values).values)

        # ---- Call the training function ----
        result_obj = train_model(
            X_train, y_train, X_test, y_test,
            epochs, lr, batch_size,
            ds_id=ds_id,
            model=model_choice
        )

        # ---- Unpack the new return shape ----
        if isinstance(result_obj, dict):
            return jsonify({
                'metrics': result_obj.get('metrics', {})
            }), 200

        # Fallback if train_model returns a plain string
        return jsonify({'classification_report': str(result_obj)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _model_key(ds_id, lr, epochs, batch_size):
    payload = json.dumps({
        "ds_id": int(ds_id),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]



@app.route('/model_results', methods=['POST'])
def model_results():
    print("Inside model_results function.")
    try:
        data = request.get_json(force=True) or {}

        def _coerce_int(key):
            v = data.get(key)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                raise ValueError(f"Missing '{key}'.")
            try:
                return int(v)
            except Exception:
                raise ValueError(f"Invalid '{key}': {v!r}")

        def _coerce_float(key):
            v = data.get(key)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                raise ValueError(f"Missing '{key}'.")
            try:
                return float(v)
            except Exception:
                raise ValueError(f"Invalid '{key}': {v!r}")

        ds_id = _coerce_int('ds_id')
        lr = _coerce_float('lr')
        epochs = _coerce_int('epochs')
        batch_size = _coerce_int('batch_size')
        model_choice = (data.get('model') or 'cnn').lower().strip()
        print("Parsed payload:", ds_id, lr, epochs, batch_size, model_choice)

        # --- Build model file path (model-specific first; fallback to legacy) ---
        model_path = os.path.join(
            "saved_models",
            f"{model_choice}_ds{ds_id}_{lr}_{epochs}_{batch_size}.keras"
        )
        if not os.path.exists(model_path):
            # Legacy filename (pre-model-choice saves)
            legacy_path = os.path.join("saved_models", f"model_ds{ds_id}_{lr}_{epochs}_{batch_size}.keras")
            if os.path.exists(legacy_path):
                model_path = legacy_path
            else:
                return jsonify({"error": f"No saved model found for these parameters."}), 404

        print("model_path:", model_path)
        print("Custom objects passed to load_model:", {"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding})

        # --- Load model ---
        #model = keras_load_model(model_path, custom_objects=None, compile=True, safe_mode=True)
        if model_choice == "transformer":

            model = keras_load_model(
                model_path,
                custom_objects={"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding},
                compile=True,
                safe_mode=True
            )
        else:
            model = keras_load_model(model_path, custom_objects=None, compile=True, safe_mode=True)
        print("Model loaded.")

        # --- Load dataset ---
        X_train_df, y_train, X_test_df, y_test = load_dataset_cached(ds_id)
        if X_train_df is None:
            return jsonify({"error": f"Dataset {ds_id} not found"}), 404
        print("Dataset loaded.")

        # --- Label encoding (fit on train+test to be safe) ---
        le = LabelEncoder()
        le.fit(np.asarray(list(y_train) + list(y_test), dtype=object))

        # --- Helpers ---
        def to_3d(df):
            return np.stack(df.iloc[:, 0].apply(lambda s: s.values).values)[..., np.newaxis]

        def to_2d(df):
            return np.stack(df.iloc[:, 0].apply(lambda s: s.values).values)

        def zscore_per_sample(X):
            X = np.asarray(X, dtype=np.float32)
            m = X.mean(axis=1, keepdims=True)
            s = X.std(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return (X - m) / s

        # --- Preprocess + Predict by dataset ---
        if ds_id == 1:
            # ItalyPowerDemand (binary)
            X_test = to_3d(X_test_df)
            # Transformer in your notebooks uses per-sample z-score
            if model_choice == 'transformer':
                X_test = zscore_per_sample(X_test)
            else:
                # If your CNN was also trained with z-score, keep this enabled
                X_test = zscore_per_sample(X_test)

            y_test_enc = le.transform(np.asarray(y_test))
            y_prob = model.predict(X_test, verbose=0)
            # Robust output handling: 1-unit (sigmoid) vs 2-unit (softmax)
            if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
                y_pred = (y_prob.ravel() > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_prob, axis=1)

        elif ds_id == 2:
            # CinCECGTorso (multiclass)
            X_test = to_3d(X_test_df)
            if model_choice == 'transformer':
                X_test = zscore_per_sample(X_test)  # match Transformer training
            # (CNN path: keep as trained; add z-score if you used it in training.)

            y_test_enc = le.transform(np.asarray(y_test))
            y_prob = model.predict(X_test, verbose=0)
            # Expecting multiclass softmax; fall back just in case
            if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
                y_pred = (y_prob.ravel() > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_prob, axis=1)

        elif ds_id == 3:
            # ECG200 (binary)
            X_test_2d = to_2d(X_test_df)
            X_test = X_test_2d.reshape((X_test_2d.shape[0], X_test_2d.shape[1], 1))
            if model_choice == 'transformer':
                X_test = zscore_per_sample(X_test)
            else:
                # If your CNN was trained with z-score, enable it here too.
                # X_test = zscore_per_sample(X_test)
                pass

            y_test_enc = le.transform(np.asarray(y_test))
            y_prob = model.predict(X_test, verbose=0)
            if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
                y_pred = (y_prob.ravel() > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_prob, axis=1)
        else:
            return jsonify({"error": f"Unknown dataset ID {ds_id}"}), 400

        # --- Metrics ---
        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_enc, y_pred, average='weighted', zero_division=0
        )
        report_str = classification_report(y_test_enc, y_pred, zero_division=0)
        print("Returning metrics.")

        return jsonify({
            "ok": True,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    os.makedirs('saved_models', exist_ok=True)
    app.run(debug=True, use_reloader=False, threaded=True)  
