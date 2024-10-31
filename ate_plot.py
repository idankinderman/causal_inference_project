import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_ATE_with_confidence_intervals(dir1, dir2, dir3, title1, title2, title3, main_title, save_path,
                                          color1='blue', color2='green'):
    # Increase the text sizes
    label_size = 16
    title_size = 20
    tick_size = 14

    # Find the global maximum value across all directories
    all_means = []
    all_highs = []
    all_lows = []

    for directory in [dir1, dir2, dir3]:
        for vals in directory.values():
            all_lows.extend([vals["# Children expected 79"][0], vals["# Children ideal 79"][0]])
            all_means.extend([vals["# Children expected 79"][1], vals["# Children ideal 79"][1]])
            all_highs.extend([vals["# Children expected 79"][2], vals["# Children ideal 79"][2]])

    max_y_value = max(all_highs)
    min_y_value = min(all_lows)

    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set the main title for the entire figure
    fig.suptitle(main_title, fontsize=title_size + 4)

    # Function to plot each subplot
    def plot_subplot(ax, directory, title):
        keys = list(directory.keys())
        x_range = range(len(keys))

        means_expected = [vals["# Children expected 79"][1] for vals in directory.values()]  # Extracting mean values
        lower_errors_expected = [vals["# Children expected 79"][1] - vals["# Children expected 79"][0] for vals in
                                 directory.values()]  # Lower errors
        upper_errors_expected = [vals["# Children expected 79"][2] - vals["# Children expected 79"][1] for vals in
                                 directory.values()]  # Upper errors
        error_bars_expected = [lower_errors_expected, upper_errors_expected]

        means_ideal = [vals["# Children ideal 79"][1] for vals in directory.values()]  # Extracting mean values
        lower_errors_ideal = [vals["# Children ideal 79"][1] - vals["# Children ideal 79"][0] for vals in
                              directory.values()]  # Lower errors
        upper_errors_ideal = [vals["# Children ideal 79"][2] - vals["# Children ideal 79"][1] for vals in
                              directory.values()]  # Upper errors
        error_bars_ideal = [lower_errors_ideal, upper_errors_ideal]

        # Offset the two series slightly on the x-axis to distinguish them
        x_offset = 0.1
        ax.errorbar([x - x_offset for x in x_range], means_expected, yerr=error_bars_expected,
                    fmt='o', color=color1, ecolor='dodgerblue', capsize=5, markersize=8, label="# Children expected 79")

        ax.errorbar([x + x_offset for x in x_range], means_ideal, yerr=error_bars_ideal,
                    fmt='o', color=color2, ecolor='limegreen', capsize=5, markersize=8, label="# Children ideal 79")

        # Set labels and title
        ax.set_title(title, fontsize=title_size)
        ax.set_ylabel('ATE', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)

        # Display strings on the x-axis
        ax.set_xticks(x_range)
        ax.set_xticklabels(keys, fontsize=tick_size, rotation=45, ha='right')

        # Set the y-axis range to be the same across all plots
        ax.set_ylim(min(0, min_y_value - 0.1), max_y_value + 0.1)  # Adding a small buffer to the max value

        ax.grid(True)
        ax.legend(fontsize=label_size - 2)  # Add legend to distinguish between the two series

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
    """
    dir1 = {'S-learner': [0.019201057866849013, 0.10748502622217614, 0.2142904936447768],
            'T-learner': [0.09968651679611173, 0.22442348072323706, 0.3476276513989843],
            'Matching': [0.042839374555792474, 0.19159192134565267, 0.3359556384742952]}

    dir2 = {'S-learner': [0.035365652090406168, 0.13208454391386297, 0.21584471212541296],
            'T-learner': [0.11897666553164755, 0.24276123068171255, 0.3837212130352622],
            'Matching': [0.13426375768481033, 0.26534143049932525, 0.39396311291048135]}

    dir3 = {'S-learner': [0.03044049992256798, 0.15000141309061785, 0.3087912877426343],
            'T-learner': [0.21993336812443282, 0.4148107888072976, 0.6227929933245071],
            'Matching': [0.24214241866175568, 0.456413546142828, 0.6953243298547166]}
    """

    # Young without children participants
    dir1 = {'S-learner': {"# Children expected 79" : [0.0088, 0.0918, 0.2093], "# Children ideal 79" : [-0.0153, 0.0609, 0.1643]},
            'T-learner': {"# Children expected 79" : [0.0494, 0.1870, 0.3444], "# Children ideal 79" : [-0.0003, 0.1497, 0.297]},
            'Matching': {"# Children expected 79" : [0.0151, 0.1835, 0.3535], "# Children ideal 79" : [-0.0118, 0.1558, 0.3199]},
            'IPW': {"# Children expected 79" : [-0.2837, 0.1745, 0.7], "# Children ideal 79" : [-0.4269, 0.1098, 0.6179]}
            }

    # Mature without children participants
    dir2 = {'S-learner': {"# Children expected 79" : [0.0421, 0.1303, 0.2398], "# Children ideal 79" : [0.0374, 0.1239, 0.2419]},
            'T-learner': {"# Children expected 79" : [0.0832, 0.2204, 0.3606], "# Children ideal 79" : [0.0753, 0.2177, 0.3631]},
            'Matching': {"# Children expected 79" : [0.1019, 0.2436, 0.3781], "# Children ideal 79" : [0.0372, 0.1760, 0.3116]},
            'IPW': {"# Children expected 79" : [-0.2032, 0.2419, 0.6878], "# Children ideal 79" : [-0.2665, 0.2064, 0.6309]}
            }

    # Mature with children participants
    dir3 = {'S-learner': {"# Children expected 79" : [0.0809, 0.2139, 0.3663], "# Children ideal 79" : [0.0908, 0.2440, 0.4487]},
            'T-learner': {"# Children expected 79" : [0.2689, 0.4791, 0.6818], "# Children ideal 79" : [0.3304, 0.5551, 0.7727]},
            'Matching': {"# Children expected 79" : [0.2729, 0.5055, 0.7539], "# Children ideal 79" : [0.2541, 0.5293, 0.7907]},
            'IPW': {"# Children expected 79" : [-0.3282, 0.4447, 1.1937], "# Children ideal 79" : [-0.3558, 0.4459, 1.2902]}
            }

    plot_ATE_with_confidence_intervals(dir1,
                                       dir2,
                                       dir3,
                                       title1="Young without children participants",
                                       title2="Mature without children participants",
                                       title3="Mature with children participants",
                                       main_title="",
                                       save_path="figures/ate_plot.png",)