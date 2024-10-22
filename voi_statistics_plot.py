import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

def select_csv_file():
    """Open a file dialog to select a CSV file and return the file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    get_data(file_path)
    return file_path

def get_data(csv_file):
    """Read the CSV file and return the data that we want to plot."""
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=';')
    # Print the columns to understand the structure
    print("Columns in the CSV file:", df.columns)
    # Get the columns that we want to plot
    df.columns = ['DiscardColumn1', 'DiscardColumn2', 'PatientName [string]',
        'DiscardColumn4', 'StudyDescription [string]', 'SeriesDescription [string]',
       'StudyDate [date_time]', 'Color [0xARGB]', 'VoiName(Region) [string]',
       'Time [seconds]', 'Averaged [g/ml]', 'Sd [g/ml]', 'Volume [ccm]',
       'Total(SUM) [g/ml]', 'Total(AVR*VOL) [({SUVbw}g/ml)*(ccm)]',
       'Min [g/ml]', 'Max [g/ml]', 'NumberOfPixels [voxels]',
       'NumberOfEffectivePixels [voxels]', 'HotAveraged(0) [g/ml]',
       'Stdv [g/ml]', 'Q1 [g/ml]', 'Median [g/ml]', 'Q3 [g/ml]',
       'AreaUnderCurve [({SUVbw}g/ml)*(seconds)]', 'HypoxiaIndex [1/1]',
       'HypoxiaVolume [ccm]']

    # Filter rows where data is 'prostate_lesion'
    prostate_lesion_data = df[df['VoiName(Region) [string]'] == 'prostate_lesion']
    #print(prostate_lesion_data)

    # Filter rows where data is 'prostate_healthy'
    prostate_healthy_data = df[df['VoiName(Region) [string]'] == 'prostate_healthy']
    #print(prostate_healthy_data)

    # Filter rows where data is 'gluteus_maximus'
    gluteus_maximus_data = df[df['VoiName(Region) [string]'] == 'gluteus_maximus']
    #print(gluteus_maximus_data)

    # Plot and save the plots
    # Prostate lesion plots
    plot_and_save_average_suv(prostate_lesion_data, csv_file, (2, 11))
    plot_and_save_snr(prostate_lesion_data, csv_file, (2, 5))
    # Healthy prostate plots
    plot_and_save_average_suv(prostate_healthy_data, csv_file, (0, 2))
    plot_and_save_snr(prostate_healthy_data, csv_file, (2, 4))
    # Gluteus maximus plots
    plot_and_save_average_suv(gluteus_maximus_data, csv_file, (0, 1))
    plot_and_save_snr(gluteus_maximus_data, csv_file, (0, 16))


def plot_and_save_average_suv(data, csv_file, yticks_range):
    """Plot the data and save the plot as PNG and pickle."""
    # Mapping for VOI names
    voi_name_mapping = {
        'prostate_lesion': 'Prostate Lesion',
        'prostate_healthy': 'Healthy Prostate',
        'gluteus_maximus': 'Gluteus Maximus'
    }

    # Get the unique value from the 'VoiName(Region) [string]' column
    voi_name_key = data['VoiName(Region) [string]'].unique()[0]
    voi_name = voi_name_mapping.get(voi_name_key, voi_name_key)
    # Get the unique value from the 'PatientName [string]' column
    patient_name = data['PatientName [string]'].unique()[0]
    print('Patient: ')
    print(patient_name)
    # Plot the 'Averaged [g/ml]' values against the iteration number with standard deviations
    plt.figure()
    plt.plot(range(1, len(data) + 1), data['Averaged [g/ml]'], marker='o')
    plt.errorbar(range(1, len(data) + 1), data['Averaged [g/ml]'],
                 yerr=data['Sd [g/ml]'], fmt='o', capsize=5, label='Averaged [g/ml]', alpha=0.7)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Averaged SUV {SUVbw} [g/ml]')
    plt.ylim(yticks_range)  # Limit the y-axis
    plt.xticks(range(1, 9))  # Set x-axis values to 1-8
    plt.title(f'{voi_name} Averaged Body Weight SUVs')
    plt.grid(True)
    #plt.legend()

    # Save the plot as PNG and pickle
    base_path = os.path.dirname(csv_file)
    png_path = os.path.join(base_path, f'{patient_name}_{voi_name_key}_Averaged_SUVs.png')
    pickle_path = os.path.join(base_path, f'{patient_name}_{voi_name_key}_Averaged_SUVs.pickle')

    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)

    plt.show()

def plot_and_save_snr(data, csv_file, yticks_range):
    """Plot the data and save the plot as PNG and pickle."""
    # Mapping for VOI names
    voi_name_mapping = {
        'prostate_lesion': 'Prostate Lesion',
        'prostate_healthy': 'Healthy Prostate',
        'gluteus_maximus': 'Gluteus Maximus'
    }

    # Get the unique value from the 'VoiName(Region) [string]' column
    voi_name_key = data['VoiName(Region) [string]'].unique()[0]
    voi_name = voi_name_mapping.get(voi_name_key, voi_name_key)
    # Get the unique value from the 'PatientName [string]' column
    patient_name = data['PatientName [string]'].unique()[0]
    print('Patient: ')
    print(patient_name)

    # Calculate the signal-to-noise ratio (SNR)
    data['SNR'] = data['Averaged [g/ml]'] / data['Sd [g/ml]']

    # Plot the 'Averaged [g/ml]' values against the iteration number with standard deviations
    plt.figure()
    plt.plot(range(1, len(data) + 1), data['SNR'], marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Averaged SUV / Standard Deviation [1]')
    plt.ylim(yticks_range)  # Limit the y-axis
    plt.xticks(range(1, 9))  # Set x-axis values to 1-8
    plt.title(f'{voi_name} Signal to Noise Ratio')
    plt.grid(True)
    #plt.legend()

    # Save the plot as PNG and pickle
    base_path = os.path.dirname(csv_file)
    png_path = os.path.join(base_path, f'{patient_name}_{voi_name_key}_SNR.png')
    pickle_path = os.path.join(base_path, f'{patient_name}_{voi_name_key}_SNR.pickle')

    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)

    plt.show()

if __name__ == "__main__":
    # Select the CSV file
    csv_file = select_csv_file()
    
    