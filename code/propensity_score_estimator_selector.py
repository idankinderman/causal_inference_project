import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, make_scorer, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import stats
import seaborn as sns  # Used for improved plotting


class PropensityScoreEstimatorSelector:
    """
    Class used to select model type for propensity score estimation with handling of calibration,
    cross-validation, and evaluation metrics.
    """

    def __init__(self, dataset_name="Dataset"):
        self.models = {}
        self.model_scores = {}
        self.default_model = None
        self.dataset_name = dataset_name
        self.X = None
        self.Y = None
        self.T = None
        self.model_list = {
            'Logistic Regression (L2)': {
                'model': LogisticRegression(max_iter=1000),
                'params': {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10]}
            },
            'Logistic Regression (ElasticNet)': {
                'model': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000),
                'params': {'l1_ratio': [0.1, 0.5, 0.9], 'C': [0.01, 0.1, 1, 10]}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3]}
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7]}
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            },
            'XGBoost': {
                'model': XGBClassifier(eval_metric='logloss', verbosity=1),
                'params': {'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [50, 100, 200]}
            }
        }

    def fit(self, X, Y, T):
        """
        Train multiple models using cross-validation with scaling, calibrate them, and store metrics from cross-validation.
        """
        self.X = X
        self.Y = Y
        self.T = T

        # Split data for calibration to avoid data leakage
        X_train, X_calib, T_train, T_calib = train_test_split(X, T, test_size=0.2, random_state=42, stratify=T)

        # Define a custom scoring dictionary including Brier score
        scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'brier': make_scorer(brier_score_loss, needs_proba=True)
        }

        # Train each model using GridSearchCV and calculate metrics using cross-validation
        for model_name, model_info in self.model_list.items():
            model = model_info['model']
            params = model_info['params']

            print(f"Training model: {model_name}")

            # Preprocess only if necessary for the model type
            if isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier)):
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]),
                         X.columns)
                    ],
                    remainder='passthrough'
                )
            else:
                preprocessor = 'passthrough'

            # Create a pipeline with scaling and the model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if params:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid={'model__' + key: value for key, value in params.items()},
                    cv=5,
                    scoring=scoring,
                    verbose=0,  # Less verbose output
                    refit='f1'
                )
                grid_search.fit(X_train, T_train)
                best_model = grid_search.best_estimator_
                cv_results = cross_validate(best_model, X_train, T_train, cv=5, scoring=scoring,
                                            return_train_score=True)
            else:
                cv_results = cross_validate(pipeline, X_train, T_train, cv=5, scoring=scoring, return_train_score=True)
                best_model = pipeline.fit(X_train, T_train)

            # Store the original (uncalibrated) model
            self.models[model_name] = best_model

            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_calib, T_calib)
            calibrated_model_name = f"{model_name} (Calibrated)"
            self.models[calibrated_model_name] = calibrated_model

            # Collect and store performance metrics
            self.model_scores[model_name] = self._extract_cv_metrics(cv_results)
            self.model_scores[calibrated_model_name] = self._evaluate_calibrated_model(calibrated_model, X_calib,
                                                                                       T_calib)

        # Select the default model with the highest F1 score
        best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['F1 Score'])
        self.default_model = self.models[best_model_name]

    def _extract_cv_metrics(self, cv_results):
        """Helper method to extract metrics from cross-validation results."""
        return {
            'Train Accuracy': np.mean(cv_results['train_accuracy']),
            'Test Accuracy': np.mean(cv_results['test_accuracy']),
            'AUC': np.mean(cv_results['test_roc_auc']),
            'F1 Score': np.mean(cv_results['test_f1']),
            'Brier Score': np.mean(cv_results['test_brier'])
        }

    def _evaluate_calibrated_model(self, calibrated_model, X, T):
        """Evaluate calibrated model and return performance metrics."""
        pred_proba = calibrated_model.predict_proba(X)[:, 1]
        pred_labels = calibrated_model.predict(X)
        return {
            'Test Accuracy': accuracy_score(T, pred_labels),
            'AUC': roc_auc_score(T, pred_proba),
            'F1 Score': f1_score(T, pred_labels),
            'Brier Score': brier_score_loss(T, pred_proba)
        }

    def set_default(self, model_name):
        """Set a specific model as the default model."""
        if model_name in self.models:
            self.default_model = self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

    def plot_calibration(self):
        """Plot calibration curves and histograms for all models, ordered by Brier score, with separate plots for training and validation sets."""

        class_proportions = np.bincount(self.T) / len(self.T)
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1]['Brier Score'])

        # Split the data into a new training and validation set
        X_train, X_val, T_train, T_val = train_test_split(self.X, self.T, test_size=0.2, random_state=42,
                                                          stratify=self.T)

        for model_name, model_info in sorted_models:
            model = self.models[model_name]
            auc = model_info['AUC']
            f1 = model_info['F1 Score']
            brier_score = model_info['Brier Score']
            train_acc = model_info.get('Train Accuracy')
            test_acc = model_info['Test Accuracy']

            # Retrain the model on the new training data to avoid data leakage
            model.fit(X_train, T_train)

            # Generate plots
            self._plot_model_results(
                model_name=model_name,
                model=model,
                auc=auc,
                f1=f1,
                train_acc=train_acc,
                test_acc=test_acc,
                brier_score=brier_score,
                X_train=X_train,
                T_train=T_train,
                X_val=X_val,
                T_val=T_val,
                class_proportions=class_proportions
            )

    def _plot_model_results(self, model_name, model, auc, f1, train_acc, test_acc, brier_score,
                            X_train, T_train, X_val, T_val, class_proportions):
        """Plot calibration curves and propensity score distributions for training and validation sets."""

        # Get predicted probabilities for training and validation sets
        pred_proba_train = model.predict_proba(X_train)[:, 1]
        pred_proba_val = model.predict_proba(X_val)[:, 1]

        # Create figure with 3 subplots: 1 for calibration curves, 2 for histograms
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Calibration curve plot for training and validation on the same axis
        fraction_of_positives_train, mean_predicted_value_train = calibration_curve(T_train, pred_proba_train,
                                                                                    n_bins=10)
        fraction_of_positives_val, mean_predicted_value_val = calibration_curve(T_val, pred_proba_val, n_bins=10)

        axes[0].plot(mean_predicted_value_train, fraction_of_positives_train, marker='o', label=f'{model_name} (Train)')
        axes[0].plot(mean_predicted_value_val, fraction_of_positives_val, marker='o',
                     label=f'{model_name} (Validation)')
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        axes[0].set_xlabel('Mean Predicted Value')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].legend()
        axes[0].set_title('Calibration Curve (Train vs. Validation)')

        # Histogram of predicted probabilities for training set
        treated_scores_train = pred_proba_train[T_train == 1]
        untreated_scores_train = pred_proba_train[T_train == 0]
        sns.histplot(treated_scores_train, bins=20, alpha=0.5, label='Treated', color='blue', ax=axes[1])
        sns.histplot(untreated_scores_train, bins=20, alpha=0.5, label='Untreated', color='green', ax=axes[1])
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].set_title('Training Set Propensity Score Distribution')

        # Histogram of predicted probabilities for validation set
        treated_scores_val = pred_proba_val[T_val == 1]
        untreated_scores_val = pred_proba_val[T_val == 0]
        sns.histplot(treated_scores_val, bins=20, alpha=0.5, label='Treated', color='blue', ax=axes[2])
        sns.histplot(untreated_scores_val, bins=20, alpha=0.5, label='Untreated', color='green', ax=axes[2])
        axes[2].set_xlabel('Propensity Score')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].set_title('Validation Set Propensity Score Distribution')

        # Title for all subplots
        title = f"{self.dataset_name} - {model_name}\nAUC: {auc:.4f}, F1: {f1:.4f}, Brier Score: {brier_score:.4f}, "
        if train_acc is not None:
            title += f'Train Acc: {train_acc:.2f}, '
        title += f'Accuracy: {test_acc:.2f}\nClass Proportions - Treated: {class_proportions[1]:.4f}, Untreated: {class_proportions[0]:.4f}'

        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
