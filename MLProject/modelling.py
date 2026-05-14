"""
modelling.py (MLProject version)
Training script untuk MLflow Project + CI workflow.
Nama: Athalie Aurora
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train California Housing Model')
    parser.add_argument('--n_estimators',  type=int,   default=200)
    parser.add_argument('--max_depth',     type=int,   default=10)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--train_path',    type=str,   default='california_housing_preprocessing/train.csv')
    parser.add_argument('--test_path',     type=str,   default='california_housing_preprocessing/test.csv')
    return parser.parse_args()


# ─────────────────────────────────────────────
# INIT MLFLOW
# ─────────────────────────────────────────────
def init_mlflow():
    dagshub_token = os.environ.get('DAGSHUB_TOKEN', '')
    if dagshub_token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'rathaavle'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        mlflow.set_tracking_uri('https://dagshub.com/rathaavle/ml-system.mlflow')
    else:
        mlflow.set_tracking_uri('mlruns')
    print('[INFO] MLflow tracking initialized.')


# ─────────────────────────────────────────────
# ARTEFAK: Actual vs Predicted
# ─────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted - CI Run')
    plt.tight_layout()
    path = 'actual_vs_predicted.png'
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ─────────────────────────────────────────────
# TRAINING LOGIC
# ─────────────────────────────────────────────
def run_training(args):
    # Load data
    train_df = pd.read_csv(args.train_path)
    test_df  = pd.read_csv(args.test_path)

    TARGET = 'MedHouseVal'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test  = test_df.drop(TARGET, axis=1)
    y_test  = test_df[TARGET]

    print(f'[INFO] Train: {X_train.shape} | Test: {X_test.shape}')

    # Log params
    mlflow.log_param('n_estimators',  args.n_estimators)
    mlflow.log_param('max_depth',     args.max_depth)
    mlflow.log_param('learning_rate', args.learning_rate)
    mlflow.log_param('model_type',    'GradientBoostingRegressor')

    # Train
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mlflow.log_metric('mse',  mse)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae',  mae)
    mlflow.log_metric('r2',   r2)
    mlflow.log_metric('mape', mape)

    print(f'[INFO] R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')

    # Log model
    mlflow.sklearn.log_model(model, artifact_path='model')

    # Artefak: Actual vs Predicted
    avp_path = plot_actual_vs_predicted(y_test.values, y_pred)
    mlflow.log_artifact(avp_path, artifact_path='plots')
    os.remove(avp_path)

    # Simpan model lokal untuk Docker
    os.makedirs('saved_model', exist_ok=True)
    joblib.dump(model, 'saved_model/model.pkl')
    mlflow.log_artifact('saved_model/model.pkl', artifact_path='saved_model')

    print('[DONE] Training selesai.')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    init_mlflow()

    # Jika sudah ada active run (dari MLflow Project), gunakan langsung
    # Jika tidak, buat run baru
    active_run = mlflow.active_run()
    if active_run:
        print(f'[INFO] Using existing MLflow run: {active_run.info.run_id}')
        run_training(args)
    else:
        with mlflow.start_run(run_name='CI_GradientBoosting'):
            run_training(args)


if __name__ == '__main__':
    main()
