import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class IPWATEEstimator:
    def __init__(self, n_bootstrap=1000, test_size=0.2, random_state=None, propensity_model=None):
        """
        Initialize the estimator with a propensity score model and bootstrap parameters.

        Parameters:
            n_bootstrap: int, number of bootstrap samples (default=1000).
            test_size: float, fraction of data to use for evaluation in each bootstrap sample (default=0.2).
            random_state: int or None, random seed for reproducibility (default=None).
            propensity_model: sklearn model, model to estimate propensity scores (default=None, uses LogisticRegression).
        """
        self.n_bootstrap = n_bootstrap
        self.test_size = test_size
        self.random_state = random_state
        self.propensity_model = propensity_model if propensity_model else LogisticRegression()
        self.bootstrap_random_states = np.random.RandomState(random_state).randint(0, 10000,
                                                                                   size=n_bootstrap) if random_state else None

    def _calculate_ate_ipw(self, T_eval, Y_eval, propensities):
        """Estimate ATE on evaluation data using IPW with observed outcomes."""
        # Separate treated and untreated
        treated_indices = T_eval == 1
        untreated_indices = T_eval == 0

        # Calculate weighted means for treated and untreated groups
        ate_treated = np.average(Y_eval[treated_indices], weights=1 / propensities[treated_indices])
        ate_untreated = np.average(Y_eval[untreated_indices], weights=1 / (1 - propensities[untreated_indices]))

        # Calculate ATE
        ate = ate_treated - ate_untreated
        return ate

    def estimate_ATE(self, X, Y, T):
        """
        Estimate the Average Treatment Effect (ATE) using IPW and bootstrap.

        Parameters:
            X: np.array, feature matrix.
            T: np.array, binary treatment assignment (1 for treated, 0 for untreated).
            Y: np.array, outcome variable.

        Returns:
            mean_ate: float, mean ATE from bootstrap samples.
            ci_95: tuple, lower and upper bounds of the 95% confidence interval.
        """
        bootstrap_ates = []

        # Run bootstrap procedure with progress bar
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrapping"):
            # Get a unique random state for each bootstrap sample if a base random state was provided
            sample_random_state = self.bootstrap_random_states[i] if self.bootstrap_random_states is not None else None

            # Bootstrap sample
            X_sample, T_sample, Y_sample = resample(X, T, Y, random_state=sample_random_state)

            # Split sample into training and evaluation sets
            X_train, X_eval, T_train, T_eval, Y_train, Y_eval = train_test_split(
                X_sample, T_sample, Y_sample, test_size=self.test_size
            )

            # Train the propensity model on the training set of the bootstrap sample
            self.propensity_model.fit(X_train, T_train)
            propensities_eval = self.propensity_model.predict_proba(X_eval)[:, 1]  # P(T=1|X) for evaluation set

            # Calculate ATE on the evaluation set using IPW
            ate = self._calculate_ate_ipw(T_eval, Y_eval, propensities_eval)
            bootstrap_ates.append(ate)

        # Calculate mean ATE and 95% confidence interval
        mean_ate = np.mean(bootstrap_ates)
        ci_95 = np.percentile(bootstrap_ates, [2.5, 97.5])

        return mean_ate, ci_95
