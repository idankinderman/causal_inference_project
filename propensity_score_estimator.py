import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy import stats


class PropensityScoreEstimator:
    def __init__(self, dataset_name="Dataset"):
        self.models = {}  # Now stores both calibrated and uncalibrated models
        self.model_scores = {}  # Stores both calibrated and uncalibrated model scores
        self.default_model = None
        self.dataset_name = dataset_name  # Store the dataset name for titles
        self.X_train = None
        self.X_test = None
        self.T_train = None
        self.T_test = None
        self.Y_train = None
        self.Y_test = None

    def fit(self, X_train, X_test, Y_train, Y_test, T_train, T_test, calibration_fraction=0.2):
        self.X_train = X_train
        self.X_test = X_test
        self.T_train = T_train
        self.T_test = T_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Split the training set into a training subset and a calibration subset
        X_train_sub, X_calib, T_train_sub, T_calib, Y_train_sub, Y_calib = train_test_split(
            X_train, T_train, Y_train, test_size=calibration_fraction, random_state=42
        )

        # Models to train with hyperparameters for cross-validation
        model_list = {
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

        # Train each model using GridSearchCV and calculate metrics
        for model_name, model_info in model_list.items():
            model = model_info['model']
            params = model_info['params']

            if params:
                # Use GridSearchCV for hyperparameter tuning and cross-validation, based on F1 score
                grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='f1', verbose=1)
                grid_search.fit(X_train_sub, T_train_sub)
                best_model = grid_search.best_estimator_
            else:
                # No cross-validation needed (for models without hyperparameters to tune)
                best_model = model.fit(X_train_sub, T_train_sub)

            # Store the original (uncalibrated) model
            self.models[model_name] = best_model

            # Calibrate the model and store it with '(Calibrated)' appended to the name
            calibrated_model_name = f"{model_name} (Calibrated)"
            calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_calib, T_calib)  # Use the calibration set instead of the test set
            self.models[calibrated_model_name] = calibrated_model

            # Predictions (uncalibrated)
            pred_proba = best_model.predict_proba(X_test)[:, 1]
            pred_test = best_model.predict(X_test)
            f1 = f1_score(T_test, pred_test)
            test_acc = accuracy_score(T_test, pred_test)
            train_acc = accuracy_score(T_train_sub, best_model.predict(X_train_sub))
            auc = roc_auc_score(T_test, pred_proba)
            ece = self._expected_calibration_error(T_test, pred_proba)
            brier_score = brier_score_loss(T_test, pred_proba)

            # Store model performance metrics
            self.model_scores[model_name] = {
                'Brier Score': brier_score,
                'ECE': ece,
                'AUC': auc,
                'F1 Score': f1,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc
            }

            # Predictions (calibrated)
            calibrated_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
            calibrated_brier_score = brier_score_loss(T_test, calibrated_pred_proba)
            calibrated_ece = self._expected_calibration_error(T_test, calibrated_pred_proba)
            calibrated_f1 = f1_score(T_test, calibrated_model.predict(X_test))

            # Store calibrated model performance metrics
            self.model_scores[calibrated_model_name] = {
                'Brier Score': calibrated_brier_score,
                'ECE': calibrated_ece,
                'AUC': auc,  # AUC doesn't change with calibration
                'F1 Score': calibrated_f1,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc
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
        """Plot calibration curves and histograms for all models, including calibrated ones."""
        class_proportions = np.bincount(self.T_test) / len(self.T_test)

        for model_name, model_info in self.model_scores.items():
            model = self.models[model_name]
            brier_score = model_info['Brier Score']
            ece = model_info['ECE']
            auc = model_info['AUC']
            f1 = model_info['F1 Score']
            train_acc = model_info['Train Accuracy']
            test_acc = model_info['Test Accuracy']

            # Pass the dataset name to the plot title
            self._plot_model_results(
                model_name=model_name,
                model=model,
                brier_score=brier_score,
                ece=ece,
                auc=auc,
                f1=f1,
                train_acc=train_acc,
                test_acc=test_acc,
                class_proportions=class_proportions,
                dataset_name=self.dataset_name
            )

    def predict(self, X):
        if self.default_model is None:
            raise ValueError("No default model is set. Please train the models using the 'fit' method.")
        return self.default_model.predict_proba(X)[:, 1]

    def estimate_ate(self, X, T, Y, alpha=0.05):
        """Estimates the Average Treatment Effect (ATE) with a confidence interval."""
        if self.default_model is None:
            raise ValueError("No default model is set. Please train the models using the 'fit' method.")

        # Predict propensity scores using the default model
        propensity_scores = self.default_model.predict_proba(X)[:, 1]

        # Calculate weights
        treated_mask = (T == 1)
        untreated_mask = (T == 0)

        treated_weights = 1 / propensity_scores[treated_mask]
        untreated_weights = 1 / (1 - propensity_scores[untreated_mask])

        # Estimate ATE using Inverse Probability of Treatment Weighting (IPTW)
        treated_outcome = np.average(Y[treated_mask], weights=treated_weights)
        untreated_outcome = np.average(Y[untreated_mask], weights=untreated_weights)

        ate = treated_outcome - untreated_outcome

        # Variance estimation for confidence interval (Normal approximation)
        treated_var = np.var(Y[treated_mask] * treated_weights) / len(treated_weights)
        untreated_var = np.var(Y[untreated_mask] * untreated_weights) / len(untreated_weights)
        se_ate = np.sqrt(treated_var + untreated_var)

        # Confidence interval for ATE
        z_value = stats.norm.ppf(1 - alpha / 2)
        ci_lower = ate - z_value * se_ate
        ci_upper = ate + z_value * se_ate

        return ate, (ci_lower, ci_upper)

    def _plot_model_results(self, model_name, model, brier_score, ece, auc, f1, train_acc, test_acc, class_proportions,
                            dataset_name):
        # Get predicted probabilities
        pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Create figure with 2 subplots: one for calibration curve, one for histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(self.T_test, pred_proba, n_bins=10)
        axes[0].plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name}')
        axes[0].plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        axes[0].set_xlabel('Mean Predicted Value')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].legend()
        axes[0].set_title('Calibration Curve')  # Add subtitle for the calibration curve

        # Plot histogram of predicted probabilities for treated and untreated groups
        treated_scores = pred_proba[self.T_test == 1]
        untreated_scores = pred_proba[self.T_test == 0]
        axes[1].hist(treated_scores, bins=20, alpha=0.5, label='Treated', color='blue')
        axes[1].hist(untreated_scores, bins=20, alpha=0.5, label='Untreated', color='green')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].set_title('Propensity Score Distribution')  # Add subtitle for the histogram

        # Joint title for both plots, including dataset name
        fig.suptitle(
            f'{dataset_name} - {model_name}\nBrier: {brier_score:.4f}, ECE: {ece:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, '
            f'Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}\n'
            f'Class Proportions - Treated: {class_proportions[1]:.4f}, Untreated: {class_proportions[0]:.4f}',
            fontsize=12)

        # Show the plots
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
        plt.show()

    def _expected_calibration_error(self, y_true, y_prob, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_assignments = np.digitize(y_prob, bin_edges, right=True) - 1

        bin_sums = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)
        bin_positives = np.zeros(n_bins)

        for b in range(n_bins):
            bin_mask = bin_assignments == b
            bin_total[b] = np.sum(bin_mask)
            bin_sums[b] = np.sum(y_prob[bin_mask])
            bin_positives[b] = np.sum(y_true[bin_mask])

        nonempty_bins = bin_total > 0
        bin_probs = bin_sums[nonempty_bins] / bin_total[nonempty_bins]
        bin_acc = bin_positives[nonempty_bins] / bin_total[nonempty_bins]

        ece = np.sum(bin_total[nonempty_bins] * np.abs(bin_acc - bin_probs)) / np.sum(bin_total[nonempty_bins])

        return ece
