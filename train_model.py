#!/usr/bin/env python3
"""
ECGWeb2 V2 - Model Training Script
Train an improved ECG classification model
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import catboost as cb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ECGModelTrainer:
    """Enhanced ECG Model Trainer"""

    def __init__(self, data_path=None):
        """Initialize trainer"""
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None

    def load_data(self, data_path=None):
        """
        Load and prepare ECG data

        Expected format: CSV with 187 columns
        - Columns 0-185: ECG signal data
        - Column 186: Label (0=Normal, 1=Abnormal)
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            raise ValueError("No data path provided")

        print(f"Loading data from {self.data_path}...")
        data = pd.read_csv(self.data_path)

        print(f"Data shape: {data.shape}")
        print(f"Columns: {len(data.columns)}")

        # Separate features and labels
        if len(data.columns) >= 187:
            X = data.iloc[:, :186].values
            y = data.iloc[:, 186].values
        else:
            # Assume last column is label
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {int(label)}: {count} ({count/len(y)*100:.2f}%)")

        return X, y

    def preprocess_data(self, X, y, test_size=0.2, random_state=42, scale=False):
        """
        Split and optionally scale data

        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed
            scale: Whether to standardize features
        """
        print(f"\nSplitting data (test_size={test_size})...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

        if scale:
            print("Scaling features...")
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, params=None, use_gpu=False):
        """
        Train CatBoost model

        Args:
            params: Model hyperparameters
            use_gpu: Whether to use GPU acceleration
        """
        if params is None:
            # Improved default parameters
            params = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 8,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'verbose': 100,
                'early_stopping_rounds': 50,
                'task_type': 'GPU' if use_gpu else 'CPU'
            }

        print(f"\nTraining CatBoost model...")
        print(f"Parameters: {params}")

        self.model = cb.CatBoostClassifier(**params)

        # Train with validation set
        eval_set = (self.X_test, self.y_test)

        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=eval_set,
            plot=False
        )

        print(f"\n✓ Model trained successfully!")
        print(f"Best iteration: {self.model.get_best_iteration()}")
        print(f"Best score: {self.model.get_best_score()}")

        return self.model

    def hyperparameter_tuning(self, use_gpu=False):
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            use_gpu: Whether to use GPU
        """
        print("\nPerforming hyperparameter tuning...")

        param_grid = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.03, 0.05],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [1, 3, 5]
        }

        base_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 0,
            'task_type': 'GPU' if use_gpu else 'CPU'
        }

        model = cb.CatBoostClassifier(**base_params)

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1 if not use_gpu else 1,
            verbose=2
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best score: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        return self.model

    def evaluate_model(self, save_plots=True):
        """
        Evaluate model performance

        Args:
            save_plots: Whether to save evaluation plots
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")

        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Abnormal']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        if save_plots:
            self._save_evaluation_plots(cm, y_pred_proba)

        # Cross-validation
        print(f"\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=5, scoring='roc_auc'
        )
        print(f"CV ROC AUC scores: {cv_scores}")
        print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }

    def _save_evaluation_plots(self, cm, y_pred_proba):
        """Save evaluation plots"""
        from sklearn.metrics import roc_curve

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('models_eval', exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models_eval/confusion_matrix_{timestamp}.png', dpi=150)
        plt.close()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'models_eval/roc_curve_{timestamp}.png', dpi=150)
        plt.close()

        # Feature Importance
        feature_importance = self.model.get_feature_importance()
        top_n = 20
        top_indices = np.argsort(feature_importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), feature_importance[top_indices])
        plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig(f'models_eval/feature_importance_{timestamp}.png', dpi=150)
        plt.close()

        print(f"\n✓ Evaluation plots saved to models_eval/")

    def save_model(self, model_path='Models/ECGModelImproved.pkl', save_scaler=False):
        """
        Save trained model

        Args:
            model_path: Path to save the model
            save_scaler: Whether to also save the scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f"\nSaving model to {model_path}...")
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved successfully!")

        if save_scaler and self.scaler is not None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Scaler saved to {scaler_path}")

        # Save model info
        info_path = model_path.replace('.pkl', '_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Model Training Info\n")
            f.write(f"="*60 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Model type: CatBoost Classifier\n")
            f.write(f"Training samples: {len(self.y_train)}\n")
            f.write(f"Test samples: {len(self.y_test)}\n")
            f.write(f"Best iteration: {self.model.get_best_iteration()}\n")
            f.write(f"\nModel Parameters:\n")
            for param, value in self.model.get_params().items():
                f.write(f"  {param}: {value}\n")

        print(f"✓ Model info saved to {info_path}")


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Train improved ECG classification model')
    parser.add_argument('--data', type=str, default='TestData/largedata.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='Models/ECGModelImproved.pkl',
                        help='Output model path')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--scale', action='store_true',
                        help='Scale features using StandardScaler')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')

    args = parser.parse_args()

    print("="*60)
    print("ECGWeb2 V2 - Model Training")
    print("="*60)

    # Initialize trainer
    trainer = ECGModelTrainer()

    # Load data
    X, y = trainer.load_data(args.data)

    # Preprocess
    trainer.preprocess_data(X, y, test_size=args.test_size, scale=args.scale)

    # Train model
    if args.tune:
        trainer.hyperparameter_tuning(use_gpu=args.gpu)
    else:
        trainer.train_model(use_gpu=args.gpu)

    # Evaluate
    metrics = trainer.evaluate_model(save_plots=True)

    # Save model
    trainer.save_model(args.output, save_scaler=args.scale)

    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
