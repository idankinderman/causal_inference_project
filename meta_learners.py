import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random
import pickle


# S-Learner
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors


random.seed(42)
np.random.seed(42)



#######################################################
# S Learner

class S_learner:
    def __init__(self, x_train, y_train, x_test, y_test, T_train, T_test, train_on_full_data=False):
        # Validate data before initializing
        inputs = [x_train, y_train, x_test, y_test, T_train, T_test]
        for dataset in inputs:
            self.check_nulls_in_dataframe(dataset)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.T_train = T_train
        self.T_test = T_test
        self.train_on_full_data = train_on_full_data
        self.models = {}
        self.measures = {}

    def fit(self, x_train=None, T_train=None, y_train=None, bootstrap=False):
        if x_train is None or T_train is None or y_train is None:
            x_train = self.x_train
            T_train = self.T_train
            y_train = self.y_train

        x_train_full = x_train.copy()
        x_train_full.loc[:, "T"] = T_train
        y_train = y_train.copy()

        if self.train_on_full_data:
            x_test_full = self.x_test.copy()
            x_test_full.loc[:, "T"] = self.T_test
            x_train_full = pd.concat([x_train_full, x_test_full], axis=0)
            y_test = self.y_test.copy()
            y_train = pd.concat([y_train, y_test], axis=0)

        if not bootstrap:
            self.models['linear_regression'] = LinearRegression()
            self.models['linear_regression'].fit(x_train_full, y_train)
            print("Fitted Linear Regression")

        self.models['svr_rbf'] = SVR(kernel='rbf')
        self.models['svr_rbf'].fit(x_train_full, y_train)
        if not bootstrap:
            print("Fitted SVR with RBF kernel")

        if not bootstrap:
            self.models['svr_poly'] = SVR(kernel='poly', degree=2)
            self.models['svr_poly'].fit(x_train_full, y_train)
            print("Fitted SVR with Polynomial kernel")

        self.models['gradient_boosting'] = GradientBoostingRegressor()
        self.models['gradient_boosting'].fit(x_train_full, y_train)
        if not bootstrap:
            print("Fitted Gradient Boosting Regressor")

        self.models['random_forest'] = RandomForestRegressor()
        self.models['random_forest'].fit(x_train_full, y_train)
        if not bootstrap:
            print("Fitted RandomForest Regressor")

        if not bootstrap:
            self.models['mlp'] = MLPRegressor(hidden_layer_sizes=(192,), max_iter=1500, activation='relu',
                                              solver='adam', learning_rate_init=0.00005, random_state=42)
            self.models['mlp'].fit(x_train_full, y_train)
            print("Fitted MLP Regressor")

    def evaluate(self):
        x_train_full = self.x_train.copy()
        x_test_full = self.x_test.copy()

        x_train_full.loc[:, "T"] = self.T_train
        x_test_full.loc[:, "T"] = self.T_test

        results = {}
        for name, model in self.models.items():
            train_pred = model.predict(x_train_full)
            test_pred = model.predict(x_test_full)
            train_mse = mean_squared_error(self.y_train, train_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            results[name] = {'Train MSE': train_mse, 'Test MSE': test_mse}

        for name, scores in results.items():
            print(f"{name}: Train MSE = {scores['Train MSE']:.4f}, Test MSE = {scores['Test MSE']:.4f}")

    def compute_ATE(self, bootstrap=False):
        x_combined = pd.concat([self.x_train, self.x_test], axis=0)
        self.compute_effect(x=x_combined, measure_name='ATE', bootstrap=bootstrap)

    def compute_ATE_final(self, list_of_models, bootstrap=False):
        final_ATE = 0
        for model in list_of_models:
            final_ATE += self.measures['ATE'][model]
        final_ATE = final_ATE / len(list_of_models)
        if not bootstrap:
            print("\nFinal ATE:", final_ATE)
        return final_ATE

    def compute_effect(self, x, measure_name, bootstrap=False):
        self.measures[measure_name] = {}

        x_with_T_zero = x.copy()
        x_with_T_zero['T'] = 0

        x_with_T_one = x.copy()
        x_with_T_one['T'] = 1

        for key in self.models:
            predictions_one = self.models[key].predict(x_with_T_one)
            predictions_zero = self.models[key].predict(x_with_T_zero)
            diffrences = predictions_one - predictions_zero
            self.measures[measure_name][key] = np.mean(diffrences)

        if not bootstrap:
            print()
            print(f"The {measure_name} are ", self.measures[measure_name])
            print()

    def bootstrap(self, num_trials=1000, alpha=0.05):
        n = len(self.x_train)
        self.ATE_list = []  # To store ATEs from each bootstrap sample

        for i in range(num_trials):
            if i in [num_trials // 50, num_trials // 5, 2 * num_trials // 5, 3 * num_trials // 5, 4 * num_trials // 5]:
                print(f"bootstrap {i}/{num_trials}")

            # Step 1: Generate bootstrap sample indices
            bootstrap_indices = np.random.choice(n, size=n, replace=True)

            # Step 2: Create bootstrap samples
            x_train_boot = self.x_train.iloc[bootstrap_indices]
            y_train_boot = self.y_train[bootstrap_indices]
            T_train_boot = self.T_train[bootstrap_indices]

            # Step 3: Train the model on bootstrap sample
            self.fit(x_train=x_train_boot, T_train=T_train_boot, y_train=y_train_boot, bootstrap=True)

            # Step 4: Compute ATE
            self.compute_ATE(bootstrap=True)
            final_ATE = self.compute_ATE_final(['svr_rbf', 'gradient_boosting', 'random_forest'], bootstrap=True)

            # Assume that compute_ATE_final updates self.final_ATE
            # Store the final ATE
            self.ATE_list.append(final_ATE)

        # Convert ATE list to a NumPy array for numerical operations
        ATE_array = np.array(self.ATE_list)

        # Step 5: Calculate mean ATE
        mean_ATE = np.mean(ATE_array)

        # Step 6: Calculate confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        CI_lower = np.percentile(ATE_array, lower_percentile)
        CI_upper = np.percentile(ATE_array, upper_percentile)

        # Step 7: Return results
        print("\nS-learner bootstrap")
        print("mean_ATE", mean_ATE, "CI_lower", CI_lower, "CI_upper", CI_upper)
        return mean_ATE, CI_lower, CI_upper

    def check_nulls_in_dataframe(self, df):
        # Check if there are any null values in the DataFrame
        if df.isnull().any().any():
            print("The DataFrame contains null values.")
            # Count of nulls in each column
            null_counts = df.isnull().sum()
            print("Count of null values in each column:")
            print(null_counts[null_counts > 0])

#######################################################
# T Learner

class T_learner:
    def __init__(self, x_train, y_train, x_test, y_test, T_train, T_test, train_on_full_data=False):
        # Validate data before initializing
        inputs = [x_train, y_train, x_test, y_test, T_train, T_test]
        for dataset in inputs:
            self.check_nulls_in_dataframe(dataset)

        treated_indices_train = T_train == 1
        treated_indices_test = T_test == 1

        # x
        self.x = {'train': {}, 'test': {}}
        self.x['train'][0] = x_train[~treated_indices_train].reset_index(drop=True)
        self.x['train'][1] = x_train[treated_indices_train].reset_index(drop=True)
        self.x['test'][0] = x_test[~treated_indices_test].reset_index(drop=True)
        self.x['test'][1] = x_test[treated_indices_test].reset_index(drop=True)

        # y
        self.y = {'train': {}, 'test': {}}
        self.y['train'][0] = y_train[~treated_indices_train].reset_index(drop=True)
        self.y['train'][1] = y_train[treated_indices_train].reset_index(drop=True)
        self.y['test'][0] = y_test[~treated_indices_test].reset_index(drop=True)
        self.y['test'][1] = y_test[treated_indices_test].reset_index(drop=True)

        # else
        self.train_on_full_data = train_on_full_data
        self.models = {0: {}, 1: {}}
        self.measures = {}

    def fit(self, x=None, y=None, bootstrap=False):
        if x is None or y is None:
            x = self.x
            y = self.y

        for T in [0, 1]:
            if not bootstrap:
                print(f"\nFitting models with T = {T}")

            x_train_full = x['train'][T].copy()
            y_train = y['train'][T].copy()

            if self.train_on_full_data:
                x_test_full = self.x['test'][T].copy()
                x_train_full = pd.concat([x_train_full, x_test_full], axis=0)
                y_test = self.y['test'][T].copy()
                y_train = pd.concat([y_train, y_test], axis=0)

            if not bootstrap:
                self.models[T]['linear_regression'] = LinearRegression()
                self.models[T]['linear_regression'].fit(x_train_full, y_train)
                print("Fitted Linear Regression")

            self.models[T]['svr_rbf'] = SVR(kernel='rbf')
            self.models[T]['svr_rbf'].fit(x_train_full, y_train)
            if not bootstrap:
                print("Fitted SVR with RBF kernel")

            if not bootstrap:
                self.models[T]['svr_poly'] = SVR(kernel='poly', degree=2)
                self.models[T]['svr_poly'].fit(x_train_full, y_train)
                print("Fitted SVR with Polynomial kernel")

            self.models[T]['gradient_boosting'] = GradientBoostingRegressor()
            self.models[T]['gradient_boosting'].fit(x_train_full, y_train)
            if not bootstrap:
                print("Fitted Gradient Boosting Regressor")

            self.models[T]['random_forest'] = RandomForestRegressor()
            self.models[T]['random_forest'].fit(x_train_full, y_train)
            if not bootstrap:
                print("Fitted RandomForest Regressor")

            if not bootstrap:
                self.models[T]['mlp'] = MLPRegressor(hidden_layer_sizes=(192,), max_iter=1500, activation='relu',
                                                     solver='adam', learning_rate_init=0.00005, random_state=42)
                self.models[T]['mlp'].fit(x_train_full, y_train)
                print("Fitted MLP Regressor")

    def evaluate(self):
        results = {0: {}, 1: {}}
        for T in [0, 1]:
            x_train_full = self.x['train'][T].copy()
            x_test_full = self.x['test'][T].copy()

            for name, model in self.models[T].items():
                train_pred = model.predict(x_train_full)
                test_pred = model.predict(x_test_full)
                train_mse = mean_squared_error(self.y['train'][T], train_pred)
                test_mse = mean_squared_error(self.y['test'][T], test_pred)
                results[T][name] = {'Train MSE': train_mse, 'Test MSE': test_mse}

        for T in [0, 1]:
            for name, scores in results[T].items():
                print(f"T: {T} | {name}: Train MSE = {scores['Train MSE']:.4f}, Test MSE = {scores['Test MSE']:.4f}")

    def compute_ATE(self, bootstrap=False):
        x_combined = pd.concat([self.x['train'][0], self.x['train'][1], self.x['test'][0], self.x['test'][1]], axis=0)
        self.compute_effect(x=x_combined, measure_name='ATE', bootstrap=bootstrap)

    def compute_ATE_final(self, list_of_models, bootstrap=False):
        final_ATE = 0
        for model in list_of_models:
            final_ATE += self.measures['ATE'][model]
        final_ATE = final_ATE / len(list_of_models)
        if not bootstrap:
            print("\nFinal ATE:", final_ATE)
        return final_ATE

    def compute_effect(self, x, measure_name, bootstrap=False):
        self.measures[measure_name] = {}

        for key in self.models[0]:
            predictions_one = self.models[1][key].predict(x)
            predictions_zero = self.models[0][key].predict(x)
            diffrences = predictions_one - predictions_zero
            # print(f"{measure_name}, diffrences for {key}:", diffrences)
            self.measures[measure_name][key] = np.mean(diffrences)

        if not bootstrap:
            print()
            print(f"The {measure_name} are ", self.measures[measure_name])
            print()

    def bootstrap(self, num_trials=1000, alpha=0.05):
        n0 = len(self.x['train'][0])
        n1 = len(self.x['train'][1])
        self.ATE_list = []  # To store ATEs from each bootstrap sample

        for i in range(num_trials):
            if i in [num_trials // 50, num_trials // 5, 2 * num_trials // 5, 3 * num_trials // 5, 4 * num_trials // 5]:
                print(f"bootstrap {i}/{num_trials}")

            # Step 1: Generate bootstrap sample indices
            bootstrap_indices0 = np.random.choice(n0, size=n0, replace=True)
            bootstrap_indices1 = np.random.choice(n1, size=n1, replace=True)

            # Step 2: Create bootstrap samples
            x_train_boot = {'train': {'0': None, '1': None}}
            y_train_boot = {'train': {'0': None, '1': None}}

            x_train_boot['train'][0] = self.x['train'][0].iloc[bootstrap_indices0]
            x_train_boot['train'][1] = self.x['train'][1].iloc[bootstrap_indices1]

            y_train_boot['train'][0] = self.y['train'][0][bootstrap_indices0]
            y_train_boot['train'][1] = self.y['train'][1][bootstrap_indices1]

            # Step 3: Train the model on bootstrap sample
            self.fit(x=x_train_boot, y=y_train_boot, bootstrap=True)

            # Step 4: Compute ATE
            self.compute_ATE(bootstrap=True)
            final_ATE = self.compute_ATE_final(['svr_rbf', 'gradient_boosting', 'random_forest'], bootstrap=True)

            # Assume that compute_ATE_final updates self.final_ATE
            # Store the final ATE
            self.ATE_list.append(final_ATE)

        # Convert ATE list to a NumPy array for numerical operations
        ATE_array = np.array(self.ATE_list)

        # Step 5: Calculate mean ATE
        mean_ATE = np.mean(ATE_array)

        # Step 6: Calculate confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        CI_lower = np.percentile(ATE_array, lower_percentile)
        CI_upper = np.percentile(ATE_array, upper_percentile)

        # Step 7: Return results
        print("\nT-learner bootstrap")
        print("mean_ATE", mean_ATE, "CI_lower", CI_lower, "CI_upper", CI_upper)
        return mean_ATE, CI_lower, CI_upper

    def check_nulls_in_dataframe(self, df):
        # Check if there are any null values in the DataFrame
        if df.isnull().any().any():
            print("The DataFrame contains null values.")
            # Count of nulls in each column
            null_counts = df.isnull().sum()
            print("Count of null values in each column:")
            print(null_counts[null_counts > 0])

#######################################################
# Matching

class Matching:
    def __init__(self, x, y, T):
        self.x = x.copy()
        self.y = y.copy()
        self.T = T.copy()
        # Splitting the data based on T
        self.x1 = x[T == 1].reset_index(drop=True)
        self.y1 = y[T == 1].reset_index(drop=True)
        self.x0 = x[T == 0].reset_index(drop=True)
        self.y0 = y[T == 0].reset_index(drop=True)

        self.ATE = {}
        self.ATT = {}

    def compute_ATE(self, k, x0=None, y0=None, x1=None, y1=None, bootstrap=False):
        # Ensure k is a positive integer
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k must be a positive integer")

        if x0 is None or y0 is None or x1 is None or y1 is None:
            x0 = self.x0
            y0 = self.y0
            x1 = self.x1
            y1 = self.y1

        # Initialize the nearest neighbors models
        nn_0 = NearestNeighbors(n_neighbors=k)
        nn_1 = NearestNeighbors(n_neighbors=k)

        nn_0.fit(x0.values)
        nn_1.fit(x1.values)

        # Calculate CAT for T=1
        CAT_1 = []
        for i, xi in x1.iterrows():
            _, indices = nn_0.kneighbors([xi.values])
            y_neighbors = y0.iloc[indices[0]]
            CAT_1.append(y1.loc[i] - y_neighbors.mean())

        # Calculate CAT for T=0
        CAT_0 = []
        for i, xi in x0.iterrows():
            _, indices = nn_1.kneighbors([xi.values])
            y_neighbors = y1.iloc[indices[0]]
            CAT_0.append(y_neighbors.mean() - y0.loc[i])

        # Calculate and return ATE
        all_CAT = CAT_1 + CAT_0
        self.ATE[k] = np.mean(all_CAT)
        if not bootstrap:
            print(f"ATE for k = {k} : {self.ATE[k]}")
        return self.ATE[k]

    def bootstrap(self, k, num_trials=1000, alpha=0.05):
        n0 = len(self.x0)
        n1 = len(self.x1)
        self.ATE_list = []  # To store ATEs from each bootstrap sample

        for i in range(num_trials):
            if i in [num_trials // 50, num_trials // 5, 2 * num_trials // 5, 3 * num_trials // 5, 4 * num_trials // 5]:
                print(f"bootstrap {i}/{num_trials}")

            # Step 1: Generate bootstrap sample indices
            bootstrap_indices0 = np.random.choice(n0, size=n0, replace=True)
            bootstrap_indices1 = np.random.choice(n1, size=n1, replace=True)

            x_train_boot_0 = self.x0.iloc[bootstrap_indices0].reset_index(drop=True)
            x_train_boot_1 = self.x1.iloc[bootstrap_indices1].reset_index(drop=True)
            y_train_boot_0 = self.y0[bootstrap_indices0].reset_index(drop=True)
            y_train_boot_1 = self.y1[bootstrap_indices1].reset_index(drop=True)

            final_ATE = self.compute_ATE(k=k, x0=x_train_boot_0, y0=y_train_boot_0, x1=x_train_boot_1, y1=y_train_boot_1, bootstrap=True)
            self.ATE_list.append(final_ATE)

        # Convert ATE list to a NumPy array for numerical operations
        ATE_array = np.array(self.ATE_list)

        # Step 5: Calculate mean ATE
        mean_ATE = np.mean(ATE_array)

        # Step 6: Calculate confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        CI_lower = np.percentile(ATE_array, lower_percentile)
        CI_upper = np.percentile(ATE_array, upper_percentile)

        # Step 7: Return results
        print("\nMatching bootstrap")
        print("mean_ATE", mean_ATE, "CI_lower", CI_lower, "CI_upper", CI_upper)
        return mean_ATE, CI_lower, CI_upper

#######################################################

def causal_experiment(df, title):
    print("\n\n-------------------", title, "-------------------\n")

    # 1. Converting T to binary
    threshold = 2
    df['T1_train'] = df['T1_train'].apply(lambda x: 0 if x <= threshold else 1)
    df['T2_train'] = df['T2_train'].apply(lambda x: 0 if x <= threshold else 1)
    df['T1_test'] = df['T1_test'].apply(lambda x: 0 if x <= threshold else 1)
    df['T2_test'] = df['T2_test'].apply(lambda x: 0 if x <= threshold else 1)

    for T_key, T_text in [('T1', 'Expected number of children'), ('T2', 'Ideal number of children')]:
        # 2. S-Learner
        print("\n\n------------------- S-Learner -------------------" + T_text)
        print(title, "\n")
        s_learner = S_learner(x_train=df['X_train_normalized'],
                              y_train=df['Y_train'],
                              x_test=df['X_test_normalized'],
                              y_test=df['Y_test'],
                              T_train=df[f'{T_key}_train'],
                              T_test=df[f'{T_key}_test'])
        s_learner.fit()
        s_learner.evaluate()
        s_learner.compute_ATE()
        ATE = s_learner.compute_ATE_final(['svr_rbf', 'gradient_boosting', 'random_forest'])
        print("\nBootstrap:")
        s_learner.bootstrap()


        # 3. T-Learner
        print("\n\n------------------- T-Learner -------------------" + T_text)
        print(title, "\n")
        t_learner = T_learner(x_train=df['X_train_normalized'],
                              y_train=df['Y_train'],
                              x_test=df['X_test_normalized'],
                              y_test=df['Y_test'],
                              T_train=df[f'{T_key}_train'],
                              T_test=df[f'{T_key}_test'])
        t_learner.fit()
        t_learner.evaluate()
        t_learner.compute_ATE()
        ATE = t_learner.compute_ATE_final(['svr_rbf', 'gradient_boosting', 'random_forest'])
        t_learner.bootstrap()

        # 4. Matching
        print("\n\n------------------- Matching -------------------" + T_text)
        print(title, "\n")
        matching = Matching(x=pd.concat([df['X_train_normalized'].reset_index(drop=True),
                                         df['X_test_normalized'].reset_index(drop=True)], axis=0),
                            y=pd.concat([df['Y_train'].reset_index(drop=True),df['Y_test'].reset_index(drop=True)],axis=0),
                            T=pd.concat([df[f'{T_key}_train'], df[f'{T_key}_test']], axis=0))
        matching.compute_ATE(1)
        matching.compute_ATE(3)
        matching.compute_ATE(5)
        matching.compute_ATE(9)
        matching.compute_ATE(15)
        matching.compute_ATE(50)
        matching.bootstrap(k=9)


#######################################################

if __name__ == '__main__':
    # 1. young_no_children
    with open('df_young_no_children_dict.pickle', 'rb') as f:
        df_young_no_children_dict = pickle.load(f)
    causal_experiment(df_young_no_children_dict, title="Young without children participants")

    # 2. mature_no_children
    with open('df_mature_no_children_dict.pickle', 'rb') as f:
        df_mature_no_children_dict = pickle.load(f)
    causal_experiment(df_mature_no_children_dict, title="Mature without children participants")


    # 3. mature_with_children
    with open('df_mature_with_children_dict.pickle', 'rb') as f:
        df_mature_with_children_dict = pickle.load(f)
    causal_experiment(df_mature_with_children_dict, title="Mature with children participants")