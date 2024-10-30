import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import sklearn
import random
import pickle
import matplotlib.image as mpimg


from causal_curve import GPS_Regressor
from causal_curve import TMLE_Regressor

random.seed(42)
np.random.seed(42)


def plot_mean_and_CI(ax, treatment, mean, lb, ub, color_mean=None, color_shading=None, label=None):
    # Plot the shaded range of the confidence intervals
    ax.fill_between(treatment, lb, ub, color=color_shading, alpha=0.3)
    # Plot the mean on top
    ax.plot(treatment, mean, color=color_mean, linewidth=0.75, label=label)

def plot_causal_curve(results_dict, fig_name, title, x_label, y_label):
    # Process the first key: '# Children expected 79'
    results_expected = results_dict['# Children expected 79']
    treat_expected = results_expected['Treatment']
    mean_expected = results_expected['Causal_Dose_Response']
    lb_expected = results_expected['Lower_CI']
    ub_expected = results_expected['Upper_CI']

    # Process the second key: '# Children ideal 79'
    results_ideal = results_dict['# Children ideal 79']
    treat_ideal = results_ideal['Treatment']
    mean_ideal = results_ideal['Causal_Dose_Response']
    lb_ideal = results_ideal['Lower_CI']
    ub_ideal = results_ideal['Upper_CI']

    # Find the minimum length between the two sets of data
    min_len = min(len(treat_expected), len(treat_ideal))

    # Truncate both datasets to the minimum length
    """
    treat_expected = treat_expected[:min_len]
    mean_expected = mean_expected[:min_len]
    lb_expected = lb_expected[:min_len]
    ub_expected = ub_expected[:min_len]

    treat_ideal = treat_ideal[:min_len]
    mean_ideal = mean_ideal[:min_len]
    lb_ideal = lb_ideal[:min_len]
    ub_ideal = ub_ideal[:min_len]
    """

    # Set plotting parameters
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [6, 5]

    # Create a single figure and axis
    fig, ax = plt.subplots()

    # Plot the data for '# Children expected 79'
    plot_mean_and_CI(ax, treat_expected, mean_expected, lb_expected, ub_expected, color_mean='b', color_shading='b', label='# children expected 79')

    # Plot the data for '# Children ideal 79'
    plot_mean_and_CI(ax, treat_ideal, mean_ideal, lb_ideal, ub_ideal, color_mean='g', color_shading='g', label='# children ideal 79')

    # Labels and title
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)

    # Customize plot appearance
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Calculate x-axis range with 5% padding
    x_min = min(min(treat_expected), min(treat_ideal))
    x_max = max(max(treat_expected), max(treat_ideal))
    x_range = x_max - x_min
    x_padding = 0.05 * x_range
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    # Calculate y-axis range based on lb and ub with 5% padding
    y_min = min(min(lb_expected), min(lb_ideal))
    y_max = max(max(ub_expected), max(ub_ideal))
    y_range = y_max - y_min
    y_padding = 0.05 * y_range
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Add a legend to distinguish between 'Expected' and 'Ideal'
    ax.legend(loc='best', fontsize=11)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    fig.savefig(f'figures/{fig_name}_causal_curve.png', bbox_inches='tight', dpi=300)

def stack_plots(plot1_path, plot2_path, plot3_path, save_path):
    # Load the images
    img1 = mpimg.imread(plot1_path)
    img2 = mpimg.imread(plot2_path)
    img3 = mpimg.imread(plot3_path)

    # Create a figure and axes to hold the 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display each image on the corresponding axis
    axes[0].imshow(img1)
    axes[0].axis('off')  # Hide the axes for a cleaner look

    axes[1].imshow(img2)
    axes[1].axis('off')

    axes[2].imshow(img3)
    axes[2].axis('off')

    # Adjust layout for clarity
    plt.tight_layout()

    # Save the combined plot to the specified path
    plt.savefig(save_path)

    # Optionally, close the plot to free memory
    plt.close(fig)


def create_causal_curves(df):
    gps_results = {}

    # T1 - # Children expected 79
    gps = GPS_Regressor()
    gps.fit(T=pd.concat([df['T1_train'], df['T1_test']], axis=0),
            X=pd.concat([df['X_train_normalized'].reset_index(drop=True), df['X_test_normalized'].reset_index(drop=True)], axis=0),
            y=pd.concat([df['Y_train'].reset_index(drop=True), df['Y_test'].reset_index(drop=True)], axis=0).astype('Float64'))
    gps_results['# Children expected 79'] = gps.calculate_CDRC(0.95)

    # T2 - # Children ideal 79
    gps = GPS_Regressor()
    gps.fit(T=pd.concat([df['T2_train'], df['T2_test']], axis=0).astype('Float64'),
            X=pd.concat([df['X_train_normalized'].reset_index(drop=True), df['X_test_normalized'].reset_index(drop=True)],axis=0),
            y=pd.concat([df['Y_train'].reset_index(drop=True), df['Y_test'].reset_index(drop=True)], axis=0).astype('Float64'))
    gps_results['# Children ideal 79'] = gps.calculate_CDRC(0.95)

    return gps_results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # df_young_no_children_dict
    with open('./preprocessed_data/df_young_no_children_dict.pickle', 'rb') as f:
        df_young_no_children_dict = pickle.load(f)

    gps_results = create_causal_curves(df_young_no_children_dict)
    plot_causal_curve(gps_results, fig_name="df_young_no_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nYoung without children participants",
                      x_label="Treatment", y_label="Num. of children")

    ########################################################################################################################
    # df_mature_no_children_dict
    with open('./preprocessed_data/df_mature_no_children_dict.pickle', 'rb') as f:
        df_mature_no_children_dict = pickle.load(f)

    gps_results = create_causal_curves(df_mature_no_children_dict)
    plot_causal_curve(gps_results, fig_name="df_mature_no_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nMature without children participants",
                      x_label="Treatment", y_label="Num. of children")

    ########################################################################################################################
    # df_mature_with_children_dict
    with open('./preprocessed_data/df_mature_with_children_dict.pickle', 'rb') as f:
        df_mature_with_children_dict = pickle.load(f)

    df_mature_with_children_dict['X_train_normalized'] = df_mature_with_children_dict['X_train_normalized'].drop("SAMPLE ID  79 INT_MIL FEMALE HISPANIC", axis=1)
    df_mature_with_children_dict['X_test_normalized'] = df_mature_with_children_dict['X_test_normalized'].drop("SAMPLE ID  79 INT_MIL FEMALE HISPANIC", axis=1)

    ###########

    # For the training data
    # Create a boolean mask where T_train <= 10
    mask_train1 = df_mature_with_children_dict['T1_train'] <= 9
    mask_train2 = df_mature_with_children_dict['T2_train'] <= 9
    mask_train = mask_train1 & mask_train2

    # Apply the mask to T_train, X_train_normalized, and Y_train
    df_mature_with_children_dict['T1_train'] = df_mature_with_children_dict['T1_train'][mask_train]
    df_mature_with_children_dict['T2_train'] = df_mature_with_children_dict['T2_train'][mask_train]
    df_mature_with_children_dict['X_train_normalized'] = df_mature_with_children_dict['X_train_normalized'][mask_train]
    df_mature_with_children_dict['Y_train'] = df_mature_with_children_dict['Y_train'][mask_train]

    # For the test data
    # Create a boolean mask where T_test <= 10
    mask_test1 = df_mature_with_children_dict['T1_test'] <= 9
    mask_test2 = df_mature_with_children_dict['T2_test'] <= 9
    mask_test = mask_test1 & mask_test2

    # Apply the mask to T_test, X_test_normalized, and Y_test
    df_mature_with_children_dict['T1_test'] = df_mature_with_children_dict['T1_test'][mask_test]
    df_mature_with_children_dict['T2_test'] = df_mature_with_children_dict['T2_test'][mask_test]
    df_mature_with_children_dict['X_test_normalized'] = df_mature_with_children_dict['X_test_normalized'][mask_test]
    df_mature_with_children_dict['Y_test'] = df_mature_with_children_dict['Y_test'][mask_test]

    ##########
    gps_results = create_causal_curves(df_mature_with_children_dict)
    plot_causal_curve(gps_results, fig_name="df_mature_with_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nMature with children participants",
                      x_label="Num. of children expected", y_label="Num. of children")

    #############################################
    stack_plots("figures/df_young_no_children_dict_causal_curve.png",
                "figures/df_mature_no_children_dict_causal_curve.png",
                "figures/df_mature_with_children_dict_causal_curve.png",
                "figures/causal_curve_stack.png")

