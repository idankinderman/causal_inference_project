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


def plot_mean_and_CI(ax, treatment, mean, lb, ub, color_mean=None, color_shading=None):
    # Plot the shaded range of the confidence intervals
    ax.fill_between(treatment, lb, ub, color=color_shading, alpha=0.3)
    # Plot the mean on top
    ax.plot(treatment, mean, color=color_mean, linewidth=0.75)

def plot_causal_curve(results_dict, fig_name, title, x_label, y_label):
    results_dict = results_dict[results_dict['Treatment'] <= 6]

    # Set plotting parameters
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [6, 5]

    # Create a single figure and axis
    fig, ax = plt.subplots()

    # Plotting quantities
    treat = results_dict['Treatment']
    mean = results_dict['Causal_Dose_Response']
    lb = results_dict['Lower_CI']
    ub = results_dict['Upper_CI']

    # Plot the data
    plot_mean_and_CI(ax, treat, mean, lb, ub, color_mean='b', color_shading='b')

    # Labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    # Add the main title
    #fig.suptitle(title, fontsize=10)

    # Customize plot appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Calculate x-axis range with 5% padding
    x_min, x_max = min(treat), max(treat)
    x_range = x_max - x_min
    x_padding = 0.05 * x_range
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    # Calculate y-axis range based on lb and ub with 5% padding
    y_min, y_max = min(lb), max(ub)
    y_range = y_max - y_min
    y_padding = 0.05 * y_range
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


    # Adjust layout to accommodate titles
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])

    # Save the figure
    fig.savefig(f'{fig_name}_causal_curve.png', bbox_inches='tight', dpi=300)

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

def plot_histograms(series1, series2, series3, title1, title2, title3, save_path, color='skyblue'):
    # Set the size of axis labels and titles
    label_size = 14
    title_size = 18
    tick_size = 12

    # Create a figure and subplots to hold the histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the histogram for each series with the same color
    axes[0].hist(series1, bins=20, color=color, edgecolor='black')
    axes[0].set_title(title1, fontsize=title_size)
    axes[0].set_xlabel('Treatment', fontsize=label_size)
    axes[0].set_ylabel('Frequency', fontsize=label_size)
    axes[0].tick_params(axis='both', labelsize=tick_size)

    axes[1].hist(series2, bins=20, color=color, edgecolor='black')
    axes[1].set_title(title2, fontsize=title_size)
    axes[1].set_xlabel('Treatment', fontsize=label_size)
    axes[1].set_ylabel('Frequency', fontsize=label_size)
    axes[1].tick_params(axis='both', labelsize=tick_size)

    axes[2].hist(series3, bins=20, color=color, edgecolor='black')
    axes[2].set_title(title3, fontsize=title_size)
    axes[2].set_xlabel('Treatment', fontsize=label_size)
    axes[2].set_ylabel('Frequency', fontsize=label_size)
    axes[2].tick_params(axis='both', labelsize=tick_size)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure to the given path
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close(fig)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # df_young_no_children_dict
    with open('df_young_no_children_dict.pickle', 'rb') as f:
        df_young_no_children_dict = pickle.load(f)

    gps = GPS_Regressor()
    gps.fit(T=pd.concat([df_young_no_children_dict['T_train'], df_young_no_children_dict['T_test']], axis=0),
            X=pd.concat([df_young_no_children_dict['X_train_normalized'].reset_index(drop=True), df_young_no_children_dict['X_test_normalized'].reset_index(drop=True)], axis=0),
            y=pd.concat([df_young_no_children_dict['Y_train'].reset_index(drop=True), df_young_no_children_dict['Y_test'].reset_index(drop=True)], axis=0).astype('Float64'))
    gps_results = gps.calculate_CDRC(0.95)
    plot_causal_curve(gps_results, fig_name="df_young_no_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nYoung without children participants",
                      x_label="Num. of children expected", y_label="Num. of children")

    ########################################################################################################################
    # df_mature_no_children_dict
    with open('df_mature_no_children_dict.pickle', 'rb') as f:
        df_mature_no_children_dict = pickle.load(f)

    gps = GPS_Regressor()
    gps.fit(T=pd.concat([df_mature_no_children_dict['T_train'], df_mature_no_children_dict['T_test']], axis=0),
            X=pd.concat([df_mature_no_children_dict['X_train_normalized'].reset_index(drop=True), df_mature_no_children_dict['X_test_normalized'].reset_index(drop=True)], axis=0),
            y=pd.concat([df_mature_no_children_dict['Y_train'].reset_index(drop=True), df_mature_no_children_dict['Y_test'].reset_index(drop=True)], axis=0).astype('Float64'))
    gps_results = gps.calculate_CDRC(0.95)
    plot_causal_curve(gps_results, fig_name="df_mature_no_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nMature without children participants",
                      x_label="Num. of children expected", y_label="Num. of children")

    ########################################################################################################################
    # df_mature_with_children_dict
    with open('df_mature_with_children_dict.pickle', 'rb') as f:
        df_mature_with_children_dict = pickle.load(f)

    df_mature_with_children_dict['X_train_normalized'] = df_mature_with_children_dict['X_train_normalized'].drop(
        "SAMPLE ID  79 INT_MIL FEMALE HISPANIC", axis=1)
    df_mature_with_children_dict['X_test_normalized'] = df_mature_with_children_dict['X_test_normalized'].drop(
        "SAMPLE ID  79 INT_MIL FEMALE HISPANIC", axis=1)

    ###########

    # For the training data
    # Create a boolean mask where T_train <= 10
    mask_train = df_mature_with_children_dict['T_train'] <= 11

    # Apply the mask to T_train, X_train_normalized, and Y_train
    df_mature_with_children_dict['T_train'] = df_mature_with_children_dict['T_train'][mask_train]
    df_mature_with_children_dict['X_train_normalized'] = df_mature_with_children_dict['X_train_normalized'][mask_train]
    df_mature_with_children_dict['Y_train'] = df_mature_with_children_dict['Y_train'][mask_train]

    # For the test data
    # Create a boolean mask where T_test <= 10
    mask_test = df_mature_with_children_dict['T_test'] <= 11

    # Apply the mask to T_test, X_test_normalized, and Y_test
    df_mature_with_children_dict['T_test'] = df_mature_with_children_dict['T_test'][mask_test]
    df_mature_with_children_dict['X_test_normalized'] = df_mature_with_children_dict['X_test_normalized'][mask_test]
    df_mature_with_children_dict['Y_test'] = df_mature_with_children_dict['Y_test'][mask_test]

    ##########

    #gps = TMLE_Regressor(random_seed=111, bandwidth=10)
    gps = GPS_Regressor()
    gps.fit(T=pd.concat([df_mature_with_children_dict['T_train'], df_mature_with_children_dict['T_test']], axis=0),
            X=pd.concat([df_mature_with_children_dict['X_train_normalized'].reset_index(drop=True), df_mature_with_children_dict['X_test_normalized'].reset_index(drop=True)], axis=0),
            y=pd.concat([df_mature_with_children_dict['Y_train'].reset_index(drop=True), df_mature_with_children_dict['Y_test'].reset_index(drop=True)], axis=0).astype('Float64'))
    gps_results = gps.calculate_CDRC(0.95)
    plot_causal_curve(gps_results, fig_name="df_mature_with_children_dict",
                      title="Causal Dose-Response Curve (with 95% CI)\nMature with children participants",
                      x_label="Num. of children expected", y_label="Num. of children")

    #############################################
    stack_plots("df_young_no_children_dict_causal_curve.png",
                "df_mature_no_children_dict_causal_curve.png",
                "df_mature_with_children_dict_causal_curve.png",
                "causal_curve_stack.png")


    plot_histograms(series1=pd.concat([df_young_no_children_dict['T_train'], df_young_no_children_dict['T_test']], axis=0),
                    series2=pd.concat([df_mature_no_children_dict['T_train'], df_mature_no_children_dict['T_test']], axis=0),
                    series3=pd.concat([df_mature_with_children_dict['T_train'], df_mature_with_children_dict['T_test']], axis=0),
                    title1="Young without children",
                    title2="Mature without children",
                    title3="Mature with children",
                    save_path="histograms.png")
