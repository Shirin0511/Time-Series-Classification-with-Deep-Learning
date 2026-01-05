# visualizations.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
import seaborn as sns


def generate_visualizations(ds_id):
    fig, ax = plt.subplots(figsize=(10, 6))

    # dataset file paths
    # load the TRAIN file for visualization purposes
    dataset_files = {
        1: 'data/ItalyPowerDemand_TRAIN.ts',
        2: 'data/CinCECGTorso_TRAIN.ts',
        3: 'data/ECG200_TRAIN.ts',

    }

    # Load the correct dataset using sktime's function
    if ds_id in dataset_files:
        try:
            X_train, y_train = load_from_tsfile_to_dataframe(dataset_files[ds_id])
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found for ID {ds_id}: {dataset_files[ds_id]}")
    else:
        raise ValueError("Invalid dataset ID provided.")

    # --- Plotting Logic for Dataset 1 (Italy) ---
    if ds_id == 1:
        unique_labels = sorted(np.unique(y_train))
        if len(unique_labels) != 2:
            raise ValueError("Expected a binary classification dataset for ID 1.")

        # Filter data for each class
        X_class_1 = X_train[y_train == unique_labels[0]].iloc[:, 0]
        X_class_2 = X_train[y_train == unique_labels[1]].iloc[:, 0]

        # Stack into 2D arrays to calculate mean and std
        stacked_class_1 = np.stack(X_class_1.values)
        stacked_class_2 = np.stack(X_class_2.values)

        mean_class_1 = np.mean(stacked_class_1, axis=0)
        std_class_1 = np.std(stacked_class_1, axis=0)
        mean_class_2 = np.mean(stacked_class_2, axis=0)
        std_class_2 = np.std(stacked_class_2, axis=0)

        # Plot Class 1
        ax.plot(mean_class_1, label=f"Mean Class {unique_labels[0]}", color='blue')
        ax.fill_between(range(len(mean_class_1)), mean_class_1 - std_class_1, mean_class_1 + std_class_1,
                        color='blue', alpha=0.2)

        # Plot Class 2
        ax.plot(mean_class_2, label=f"Mean Class {unique_labels[1]}", color='red')
        ax.fill_between(range(len(mean_class_2)), mean_class_2 - std_class_2, mean_class_2 + std_class_2,
                        color='red', alpha=0.2)

        ax.set_title("Average Time Series of Each Class with Standard Deviation")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

     # Plotting for ECGTorso Dataset
    elif ds_id == 2:
        class_labels = sorted(np.unique(y_train))

        # Define a color palette for the classes
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_labels)))

        for i, label in enumerate(class_labels):
            # Find all series belonging to the current class
            # This assumes your data has only one dimension, 'dim_0'
            class_series = X_train[y_train == label].iloc[:, 0]

            # Stack them into a 2D numpy array to calculate mean and std
            stacked_series = np.stack(class_series.values)

            mean_series = np.mean(stacked_series, axis=0)
            std_series = np.std(stacked_series, axis=0)

            # Plot the mean series
            ax.plot(mean_series, label=f'Class {label}', color=colors[i], linewidth=2)

            # Plot the standard deviation as a shaded area
            ax.fill_between(range(len(mean_series)), mean_series - std_series, mean_series + std_series,
                            color=colors[i], alpha=0.2)

        ax.set_title('Average Time Series per Class (with ±1 std dev)', fontsize=18)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Average Amplitude', fontsize=12)
        ax.legend(title='Class', fontsize=12)

    # --- Plotting Logic for Dataset 3 (ECG200) ---
    elif ds_id == 3:
        # --- Data Preparation ---
        X_train_np = np.stack(X_train['dim_0'].values)
        y_train_int = y_train.astype(int)

        # Separate the data by class
        class_1_data = X_train_np[y_train_int == 1]
        class_neg1_data = X_train_np[y_train_int == -1]

        # Calculate mean and standard deviation for each class
        mean_class_1 = class_1_data.mean(axis=0)
        std_class_1 = class_1_data.std(axis=0)

        mean_class_neg1 = class_neg1_data.mean(axis=0)
        std_class_neg1 = class_neg1_data.std(axis=0)

        # Time axis for plotting
        time_steps = np.arange(X_train_np.shape[1])

        # --- Visualization ---
        # Note: Using the pre-defined `ax` object
        sns.set_style("whitegrid")

        # Plot for Class 1
        ax.plot(time_steps, mean_class_1, label='Mean - Class 1 (Normal)', color='dodgerblue', linewidth=2.5)
        ax.fill_between(time_steps, mean_class_1 - std_class_1, mean_class_1 + std_class_1, color='dodgerblue',
                        alpha=0.2, label='Std. Dev - Class 1')

        # Plot for Class -1
        ax.plot(time_steps, mean_class_neg1, label='Mean - Class -1 (MI)', color='orangered', linewidth=2.5)
        ax.fill_between(time_steps, mean_class_neg1 - std_class_neg1, mean_class_neg1 + std_class_neg1,
                        color='orangered', alpha=0.2, label='Std. Dev - Class -1')

        ax.set_title('Average ECG Waveform by Class with Standard Deviation', fontsize=16, weight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.legend(loc='upper right')

    else:
        # This part should not be reached if the ID is handled above
        raise ValueError("Invalid dataset ID provided.")

    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free up memory

    return buf