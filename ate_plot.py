import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_ATE_with_confidence_intervals(dir1, dir2, dir3, title1, title2, title3, main_title, save_path, color='blue'):
    # Increase the text sizes
    label_size = 16
    title_size = 20
    tick_size = 14

    # Find the global maximum value across all directories
    all_means = [vals[1] for vals in dir1.values()] + [vals[1] for vals in dir2.values()] + [vals[1] for vals in
                                                                                             dir3.values()]
    all_highs = [vals[2] for vals in dir1.values()] + [vals[2] for vals in dir2.values()] + [vals[2] for vals in
                                                                                             dir3.values()]
    max_y_value = max(all_highs)

    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set the main title for the entire figure
    fig.suptitle(main_title, fontsize=title_size + 4)

    # Function to plot each subplot
    def plot_subplot(ax, directory, title):
        keys = list(directory.keys())
        means = [vals[1] for vals in directory.values()]  # Extracting mean values
        lower_errors = [vals[1] - vals[0] for vals in directory.values()]  # Calculating lower errors
        upper_errors = [vals[2] - vals[1] for vals in directory.values()]  # Calculating upper errors
        error_bars = [lower_errors, upper_errors]

        # Scatter plot for mean values with error bars
        ax.errorbar(keys, means, yerr=error_bars, fmt='o', color=color, ecolor='purple', capsize=5, markersize=8)

        # Adding text (mean values) above each point
        """
        for i, key in enumerate(keys):
            ax.text(i, means[i] + 0.05, f'{means[i]:.2f}', ha='center', fontsize=tick_size)
        """

        # Set labels and title
        ax.set_title(title, fontsize=title_size)
        #ax.set_xlabel('Keys', fontsize=label_size)
        ax.set_ylabel('ATE', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)

        # Display strings on the x-axis
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, fontsize=tick_size, rotation=45, ha='right')

        # Set the y-axis range to be the same across all plots
        ax.set_ylim(0, max_y_value + 0.1)  # Adding a small buffer to the max value

        ax.grid(True)

    # Plot each directory in its own subplot
    plot_subplot(axes[0], dir1, title1)
    plot_subplot(axes[1], dir2, title2)
    plot_subplot(axes[2], dir3, title3)

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure to the specified path
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close(fig)

#######################################################

if __name__ == '__main__':
    dir1 = {'S-learner': [0.019201057866849013, 0.10748502622217614, 0.2142904936447768],
            'T-learner': [0.09968651679611173, 0.22442348072323706, 0.3476276513989843],
            'Matching': [0.042839374555792474, 0.19159192134565267, 0.3359556384742952]}

    dir2 = {'S-learner': [0.035365652090406168, 0.13208454391386297, 0.21584471212541296],
            'T-learner': [0.11897666553164755, 0.24276123068171255, 0.3837212130352622],
            'Matching': [0.13426375768481033, 0.26534143049932525, 0.39396311291048135]}

    dir3 = {'S-learner': [0.03044049992256798, 0.15000141309061785, 0.3087912877426343],
            'T-learner': [0.21993336812443282, 0.4148107888072976, 0.6227929933245071],
            'Matching': [0.24214241866175568, 0.456413546142828, 0.6953243298547166]}

    plot_ATE_with_confidence_intervals(dir1,
                                       dir2,
                                       dir3,
                                       title1="Young without children participants",
                                       title2="Mature without children participants",
                                       title3="Mature with children participants",
                                       main_title="ATE with Confidence Intervals",
                                       save_path="ate_plot.png",)