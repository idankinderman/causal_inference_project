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
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats


class PropensityScoreEstimator:
    def __init__(self, dataset_name="Dataset"):
        self.models = {}  # Store trained models
        self.model_scores = {}  # Store performance metrics for models
        self.default_model = None
        self.dataset_name = dataset_name  # Store the dataset name for titles
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
            'Logistic Regression (No Penalty)': {
                'model': LogisticRegression(penalty=None, max_iter=1000),
                'params': {}
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
                'params': {}  # No hyperparameters to tune for Naive Bayes
            },
            'XGBoost': {
                'model': XGBClassifier(eval_metric='logloss', verbosity=1),
                'params': {'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [50, 100, 200]}
            }
        }

    def fit(self, X, Y, T):
        """Train multiple models using cross-validation with scaling, calibrate them, and store metrics from cross-validation."""
        self.X = X
        self.Y = Y
        self.T = T

        # Define a custom scoring dictionary including Brier score
        scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'brier': make_scorer(brier_score_loss, response_method='predict_proba')
        }

        # Train each model using GridSearchCV and calculate metrics using cross-validation
        for model_name, model_info in self.model_list.items():
            model = model_info['model']
            params = model_info['params']

            # Print which model is being trained
            print(f"Training model: {model_name}")

            # Create a pipeline with scaling and the model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Feature scaling
                ('model', model)  # Model to train
            ])

            if params:
                # Use GridSearchCV for hyperparameter tuning and cross-validation
                grid_search = GridSearchCV(pipeline,
                                           param_grid={'model__' + key: value for key, value in params.items()},
                                           cv=5, scoring=scoring, verbose=1, refit='f1')
                grid_search.fit(X, T)
                best_model = grid_search.best_estimator_

                # Extract cross-validation results for train/test accuracy, F1, AUC, and Brier score
                cv_results = cross_validate(best_model, X, T, cv=5, scoring=scoring, return_train_score=True)
            else:
                # No cross-validation needed (for models without hyperparameters to tune)
                cv_results = cross_validate(pipeline, X, T, cv=5, scoring=scoring, return_train_score=True)
                best_model = pipeline.fit(X, T)

            # Store the original (uncalibrated) model
            self.models[model_name] = best_model

            # Calibrate the model
            calibrated_model_name = f"{model_name} (Calibrated)"
            calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X, T)
            self.models[calibrated_model_name] = calibrated_model

            # Store performance metrics from cross-validation for uncalibrated models
            train_acc_mean = np.mean(cv_results['train_accuracy'])
            test_acc_mean = np.mean(cv_results['test_accuracy'])
            auc_mean = np.mean(cv_results['test_roc_auc'])
            f1_mean = np.mean(cv_results['test_f1'])
            brier_mean = np.mean(cv_results['test_brier'])

            self.model_scores[model_name] = {
                'Train Accuracy': train_acc_mean,
                'Test Accuracy': test_acc_mean,
                'AUC': auc_mean,
                'F1 Score': f1_mean,
                'Brier Score': brier_mean
            }

            # Evaluate the calibrated model directly on the entire dataset
            pred_proba = calibrated_model.predict_proba(X)[:, 1]
            pred_labels = calibrated_model.predict(X)

            # Metrics for the calibrated model
            calibrated_test_acc = accuracy_score(T, pred_labels)
            calibrated_auc = roc_auc_score(T, pred_proba)
            calibrated_f1 = f1_score(T, pred_labels)
            calibrated_brier = brier_score_loss(T, pred_proba)

            # Store metrics for calibrated models
            self.model_scores[calibrated_model_name] = {
                'Test Accuracy': calibrated_test_acc,
                'AUC': calibrated_auc,
                'F1 Score': calibrated_f1,
                'Brier Score': calibrated_brier
            }

        # Set the default model to the one with the best F1 score (uncalibrated)
        best_model_name = min(self.model_scores, key=lambda x: self.model_scores[x]['F1 Score'])
        self.default_model = self.models[best_model_name]

    def set_default(self, model_name):
        if model_name in self.models:
            self.default_model = self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

    def plot_calibration(self):
        """Plot calibration curves and histograms for all models, including calibrated ones, ordered by Brier score."""
        class_proportions = np.bincount(self.T) / len(self.T)

        # Sort models by increasing Brier score
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1]['Brier Score'])

        for model_name, model_info in sorted_models:
            model = self.models[model_name]
            auc = model_info['AUC']
            f1 = model_info['F1 Score']
            brier_score = model_info['Brier Score']

            if 'Train Accuracy' in model_info:
                train_acc = model_info['Train Accuracy']
                test_acc = model_info['Test Accuracy']
                # Plot for uncalibrated models with train/test accuracy
                self._plot_model_results(
                    model_name=model_name,
                    model=model,
                    auc=auc,
                    f1=f1,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    brier_score=brier_score,
                    class_proportions=class_proportions,
                    dataset_name=self.dataset_name
                )
            else:
                # Plot for calibrated models (use "Calibration Accuracy")
                calibration_acc = model_info['Test Accuracy']
                self._plot_model_results(
                    model_name=model_name,
                    model=model,
                    auc=auc,
                    f1=f1,
                    train_acc=None,  # No train accuracy for calibrated models
                    test_acc=calibration_acc,  # Renamed to "calibration accuracy"
                    brier_score=brier_score,
                    class_proportions=class_proportions,
                    dataset_name=self.dataset_name
                )

    def predict(self, X):
        if self.default_model is None:
            raise ValueError("No default model is set. Please train the models using the 'fit' method.")
        return self.default_model.predict_proba(X)[:, 1]

    def estimate_ate(self, X, T, Y, alpha=0.05, n_bootstraps=1000, return_bootstrap_samples=False):
        """
        Estimate the ATE using bootstrapping, returning the mean ATE,
        its confidence interval, and standard error.

        Optionally return the full set of bootstrap ATE estimates.
        """
        ate_list = []

        for i in range(n_bootstraps):
            # Bootstrap resampling
            X_resampled, T_resampled, Y_resampled = resample(X, T, Y)

            # Re-train the default model on the resampled data
            model_info = self.model_list[self.default_model.named_steps['model'].__class__.__name__]
            model = model_info['model']
            params = model_info['params']

            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

            if params:
                grid_search = GridSearchCV(pipeline,
                                           param_grid={'model__' + key: value for key, value in params.items()}, cv=5,
                                           scoring='f1', verbose=0)
                grid_search.fit(X_resampled, T_resampled)
                best_model = grid_search.best_estimator_
            else:
                best_model = pipeline.fit(X_resampled, T_resampled)

            # Predict propensity scores using the retrained model
            propensity_scores = best_model.predict_proba(X_resampled)[:, 1]

            # Calculate weights
            treated_mask = (T_resampled == 1)
            untreated_mask = (T_resampled == 0)

            treated_weights = 1 / propensity_scores[treated_mask]
            untreated_weights = 1 / (1 - propensity_scores[untreated_mask])

            # Estimate ATE using Inverse Probability of Treatment Weighting (IPTW)
            treated_outcome = np.average(Y_resampled[treated_mask], weights=treated_weights)
            untreated_outcome = np.average(Y_resampled[untreated_mask], weights=untreated_weights)

            ate = treated_outcome - untreated_outcome
            ate_list.append(ate)

        # Estimate the mean ATE, standard error, and confidence intervals
        ate_mean = np.mean(ate_list)
        ate_se = np.std(ate_list)  # Standard error is the standard deviation of the bootstrapped ATE estimates
        ate_ci_lower = np.percentile(ate_list, (alpha / 2) * 100)
        ate_ci_upper = np.percentile(ate_list, (1 - alpha / 2) * 100)

        if return_bootstrap_samples:
            return ate_mean, (ate_ci_lower, ate_ci_upper), ate_se, ate_list
        else:
            return ate_mean, (ate_ci_lower, ate_ci_upper), ate_se

    def _estimate_single_ate(self, X, T, Y):
        """
        Helper function to estimate ATE for a single treatment (T).
        This is a simplified version of the ATE estimation for binary treatments.
        """
        # Re-train the default model
        model_info = self.model_list[self.default_model.named_steps['model'].__class__.__name__]
        model = model_info['model']
        params = model_info['params']

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

        if params:
            grid_search = GridSearchCV(pipeline, param_grid={'model__' + key: value for key, value in params.items()},
                                       cv=5, scoring='f1', verbose=0)
            grid_search.fit(X, T)
            best_model = grid_search.best_estimator_
        else:
            best_model = pipeline.fit(X, T)

        # Predict propensity scores using the retrained model
        propensity_scores = best_model.predict_proba(X)[:, 1]

        # Calculate weights
        treated_mask = (T == 1)
        untreated_mask = (T == 0)

        treated_weights = 1 / propensity_scores[treated_mask]
        untreated_weights = 1 / (1 - propensity_scores[untreated_mask])

        # Estimate ATE using Inverse Probability of Treatment Weighting (IPTW)
        treated_outcome = np.average(Y[treated_mask], weights=treated_weights)
        untreated_outcome = np.average(Y[untreated_mask], weights=untreated_weights)

        return treated_outcome - untreated_outcome

    def estimate_ate_difference(self, X, T1, T2, Y, alpha=0.05, n_bootstraps=1000, return_bootstrap_samples=False):
        """
        Estimate the difference in the ATE between two treatments (T1 and T2).

        Parameters:
        - X: Feature matrix
        - T1: Binary treatment assignment for T1 (1 for T1, 0 otherwise)
        - T2: Binary treatment assignment for T2 (1 for T2, 0 otherwise)
        - Y: Outcome variable
        - alpha: Confidence level (default 0.05 for 95% CI)
        - n_bootstraps: Number of bootstrap samples (default 1000)
        - return_bootstrap_samples: If True, return all bootstrap ATE estimates (default False)

        Returns:
        - ATE difference (T1 - T2)
        - Confidence interval (lower, upper)
        - Standard error of the ATE difference
        - (Optional) List of bootstrap ATE differences (if return_bootstrap_samples=True)
        """
        ate_diff_list = []

        for i in range(n_bootstraps):
            # Bootstrap resampling
            X_resampled, T1_resampled, T2_resampled, Y_resampled = resample(X, T1, T2, Y)

            # Estimate ATE for T1
            ate_T1 = self._estimate_single_ate(X_resampled, T1_resampled, Y_resampled)

            # Estimate ATE for T2
            ate_T2 = self._estimate_single_ate(X_resampled, T2_resampled, Y_resampled)

            # Calculate the difference in ATEs (T1 - T2)
            ate_diff = ate_T1 - ate_T2
            ate_diff_list.append(ate_diff)

        # Estimate the mean difference, standard error, and confidence intervals
        ate_diff_mean = np.mean(ate_diff_list)
        ate_diff_se = np.std(ate_diff_list)
        ate_diff_ci_lower = np.percentile(ate_diff_list, (alpha / 2) * 100)
        ate_diff_ci_upper = np.percentile(ate_diff_list, (1 - alpha / 2) * 100)

        if return_bootstrap_samples:
            return ate_diff_mean, (ate_diff_ci_lower, ate_diff_ci_upper), ate_diff_se, ate_diff_list
        else:
            return ate_diff_mean, (ate_diff_ci_lower, ate_diff_ci_upper), ate_diff_se

    def _plot_model_results(self, model_name, model, auc, f1, train_acc, test_acc, brier_score, class_proportions,
                            dataset_name):
        # Get predicted probabilities
        pred_proba = model.predict_proba(self.X)[:, 1]

        # Create figure with 2 subplots: one for calibration curve, one for histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(self.T, pred_proba, n_bins=10)
        axes[0].plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name}')
        axes[0].plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        axes[0].set_xlabel('Mean Predicted Value')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].legend()
        axes[0].set_title('Calibration Curve')  # Add subtitle for the calibration curve

        # Plot histogram of predicted probabilities for treated and untreated groups
        treated_scores = pred_proba[self.T == 1]
        untreated_scores = pred_proba[self.T == 0]
        axes[1].hist(treated_scores, bins=20, alpha=0.5, label='Treated', color='blue')
        axes[1].hist(untreated_scores, bins=20, alpha=0.5, label='Untreated', color='green')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].set_title('Propensity Score Distribution')  # Add subtitle for the histogram

        # Joint title for both plots, including dataset name, Brier score, and class proportions
        title = f'{dataset_name} - {model_name}\nAUC: {auc:.4f}, F1: {f1:.4f}, Brier Score: {brier_score:.4f}, '
        if train_acc is not None:
            title += f'Train Acc: {train_acc:.2f}, '
        title += f'Accuracy: {test_acc:.2f}\nClass Proportions - Treated: {class_proportions[1]:.4f}, Untreated: {class_proportions[0]:.4f}'

        fig.suptitle(title, fontsize=12)

        # Show the plots
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
        plt.show()
