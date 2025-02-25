import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from tkinter import filedialog, messagebox
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

def select_csv_file():
    """Open a file dialog to select a CSV file and return the file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select IDIF file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    get_data_csv(file_path)
    return file_path

def get_data_csv(csv_file):
    """Read the CSV file and return the data that we want to plot."""
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=';')
    print("data:\n", df)
    # Convert the time from seconds to minutes
    df['Time [minutes]'] = df['Time [seconds]'] / 60

    # Print the columns to understand the structure
    print("Columns in the CSV file:", df.columns)

    # Get the columns that we want to plot
    df.columns = ['DiscardColumn1', 'DiscardColumn2', 'PatientName [string]',
        'DiscardColumn4', 'StudyDescription [string]', 'SeriesDescription [string]',
       'StudyDate [date_time]', 'Color [0xARGB]', 'VoiName(Region) [string]',
       'Time [seconds]', 'Averaged [SUV]', 'Sd [SUV]', 'Volume [ccm]',
       'Total(SUM) [SUV]', 'Total(AVR*VOL) [({SUVbw}SUV)*(ccm)]',
       'Min [SUV]', 'Max [SUV]', 'NumberOfPixels [voxels]',
       'NumberOfEffectivePixels [voxels]', 'HotAveraged(0) [SUV]',
       'Stdv [SUV]', 'Q1 [SUV]', 'Median [SUV]', 'Q3 [SUV]',
       'AreaUnderCurve [({SUVbw}SUV)*(seconds)]', 'HypoxiaIndex [1/1]',
       'HypoxiaVolume [ccm]', 'Time [minutes]']

    # Filter rows where data is 'prostate_lesion'
    prostate_lesion_data = df[df['VoiName(Region) [string]'] == 'prostate_lesion']
    #print(prostate_lesion_data)

    # Filter rows where data is 'prostate_healthy'
    prostate_healthy_data = df[df['VoiName(Region) [string]'] == 'prostate_healthy']
    #print(prostate_healthy_data)

    # Filter rows where data is 'gluteus_maximus'
    gluteus_maximus_data = df[df['VoiName(Region) [string]'] == 'gluteus_maximus']
    #print(gluteus_maximus_data)

    IDIF_data = df[df['VoiName(Region) [string]'] == 'Four hottest pixels per slice']
    print("data:\n",IDIF_data)
    # Plot and save the plots
    # Prostate lesion plots
    plot_and_save_average_suv(IDIF_data, csv_file, (0, 40))
    #plot_and_save_snr(prostate_lesion_data, csv_file, (2, 4))
    # Healthy prostate plots
    #plot_and_save_average_suv(prostate_healthy_data, csv_file, (0, 2))
    #plot_and_save_snr(prostate_healthy_data, csv_file, (4, 6))
    # Gluteus maximus plots
    #plot_and_save_average_suv(gluteus_maximus_data, csv_file, (0, 1))
    #plot_and_save_snr(gluteus_maximus_data, csv_file, (0, 16))


def select_txt_file():
    """Open a file dialog to select a TXT file and return the file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select AIF file",
        filetypes=(("TXT files", "*.txt"), ("All files", "*.*"))
    )
    df = get_data_txt(file_path)
    return df

def get_data_txt(txt_file):
    """Read the TXT file and return the data that we want to plot."""
    # Read the TXT file
    df = pd.read_csv(txt_file, sep='\t')

    # Print the columns to understand the structure
    print("Columns in the TXT file:", df.columns)

    return df

def plot_and_save_average_suv(data, csv_file, yticks_range):
    """Plot the data and save the plot as PNG and pickle."""
    # Mapping for VOI names
    voi_name_mapping = {
        'prostate_lesion': 'Prostate Lesion',
        'prostate_healthy': 'Healthy Prostate',
        'gluteus_maximus': 'Gluteus Maximus',
        'Four hottest pixels per slice': 'Mourik\'s method IDIF'
    }
    # Reset the index of the DataFrame
    data = data.reset_index(drop=True)
    print(data['Time [minutes]'])
    # Get the unique value from the 'VoiName(Region) [string]' column
    voi_name_key = data['VoiName(Region) [string]'].unique()[0]
    voi_name = voi_name_mapping.get(voi_name_key, voi_name_key)
    # Get the unique value from the 'PatientName [string]' column
    patient_name = data['PatientName [string]'].unique()[0]
    print('Patient: ')
    print(patient_name)
    # Plot the 'Averaged [SUV]' values against the iteration number with standard deviations
    plt.figure()
    plt.plot(data['Time [minutes]'], data['Averaged [SUV]'], marker='o')
    #plt.errorbar(data['Time [minutes]'], data['Averaged [SUV]'], #range(1, len(data) + 1)
    #             yerr=data['Sd [SUV]'], fmt='o', capsize=5, label='Averaged [SUV]', alpha=0.7)
    plt.xlabel('Time [minutes]')
    plt.ylabel('Averaged SUV {SUVbw} [(kg*mL)$^{-1}$]')
    plt.ylim(yticks_range)  # Limit the y-axis
    #plt.xticks(range(1, 9))  # Set x-axis values to 1-8
    plt.title(f'{voi_name} Averaged Body Weight SUVs')
    plt.grid(True)
    #plt.legend()
    plt.show(block=False)
    
    # Now include the AIF data in the IDIF plot
    AIF_data = select_txt_file()
    plt.plot(AIF_data['Time [min]'], AIF_data['AIF shifted [SUV]'], marker='x', color='red')
    plt.legend(['IDIF', 'AIF'])
    plt.show(block=False)
    
    # Perform the RC correction
    rc_correction = float(input("Enter the RC correction value, [0-1]: "))
    data['Averaged [SUV]'] = data['Averaged [SUV]'] / rc_correction
    
    # Plot the corrected data
    plt.plot(data['Time [minutes]'], data['Averaged [SUV]'], marker='o')
    plt.legend(['IDIF', 'AIF', 'RC Corrected IDIF'])
    plt.ylim([0, 80])
    plt.show(block=False)

    # Save the plot as PNG, PDF and pickle
    save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//VOI Statistics//PSMA007"
    png_path = os.path.join(save_path, f'test{patient_name}_{voi_name_key}_AIF_and_IDIF_RC_correction_with_c_4_hottest_and_no_background.png')
    pickle_path = os.path.join(save_path, f'test{patient_name}_{voi_name_key}_AIF_and_IDIF_RC_correction_with_c_4_hottest_and_no_background.pickle')
    pdf_path = os.path.join(save_path, f'test{patient_name}_{voi_name_key}_AIF_and_IDIF_RC_correction_with_c_4_hottest_and_no_background.pdf')
    answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here:\n{save_path}\nas\n{png_path}?")
    if answer:
        # Save the plot as PNG, PDF, and pickle files
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(plt.gcf(), f)

    # Analyze the frequencies of the AIF data
    #fourier_transform(AIF_data)

def fourier_transform(df):
   # Extract time and signal
    time = df["Time [min]"].to_numpy()
    signal = df["AIF shifted [SUV]"].to_numpy()
    time_minutes = time * 60
    # Round the time values (they are until 691 seconds in 1 seconds spacing but the swisstrace saved them sometimes as 2.0x seconds instead of exactly 2 sec)
    time_minutes = np.round(time_minutes)

    # Cut off the time and signal after the 691st entry

    time_minutes = time_minutes[0:692]
    signal = signal[:len(time_minutes)]
    
    # Perform FFT
    N = len(signal)  # Number of data points
    T = 1.0  # Sampling interval (1 second)
    fft_values = fft(signal)  # FFT values
    frequencies = fftfreq(N, d=T)  # Frequency bins

    # Only positive frequencies (FFT output is symmetric for real inputs)
    positive_freqs = frequencies[:N // 2]
    positive_magnitudes = np.abs(fft_values[:N // 2])

    # Plot the FFT results
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitudes, color='blue', label="FFT Magnitude")
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
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
    data['SNR'] = data['Averaged [SUV]'] / data['Sd [SUV]']

    # Plot the 'Averaged [SUV]' values against the iteration number with standard deviations
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
    
    