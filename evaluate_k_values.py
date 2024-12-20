import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def process_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Select the relevant columns
    columns_to_select = ["Patient_ID", "Model", "K1", "k2", "k3", "k4"]
    selected_df = df[columns_to_select]
    
    # Filter unique rows based on 'Patient_ID' and 'Model'
    grouped = selected_df.groupby(['Patient_ID', 'Model'])
    
    # Create a dictionary to store DataFrames for each unique Patient_ID and Model
    unique_dataframes = {}
    
    for (patient_id, model), group in grouped:
        # Store the DataFrame in the dictionary
        unique_dataframes[(patient_id, model)] = group
        
    return unique_dataframes

def calculate_spearman_correlation(df1, df2):
    """
    Calculates the Spearman correlation coefficient and p-value between K1 values of two DataFrames.
    """
    k1_values_1 = df1['K1']
    k1_values_2 = df2['K1']
    
    # Ensure the lengths are the same for correlation calculation
    if len(k1_values_1) != len(k1_values_2):
        raise ValueError("The two dataframes have different lengths, cannot calculate Spearman correlation.")
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(k1_values_1, k1_values_2)
    return correlation, p_value

def plot_correlation(df1, df2, correlation, p_value):
    """
    Plots a scatter plot of K1 values from two DataFrames and annotates it with Spearman correlation and p-value.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df1['K1'], df2['K1'], alpha=0.7)
    plt.title(f"Spearman Correlation: {correlation:.2f}, P-value: {p_value:.2e}")
    plt.xlabel("K1 Values (PSMA001)")
    plt.ylabel("K1 Values (PSMA001_a)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = "your_file.csv"
    unique_dfs = process_csv(file_path)
    
    # Get DataFrames for specific Patient_ID and Model
    df_psma001_02_two_c_irre = unique_dfs.get(("PSMA001_02", "2_Tissue_Compartments"))    # two comparments irreversible
    df_psma001_02_two_c_re = unique_dfs.get(("PSMA001_02", "2_Tissue_Compartments,_FDG"))      # two compartments reversible

    if df_psma001_02_two_c_irre is not None and df_psma001_02_two_c_re is not None:
        # Calculate Spearman correlation
        try:
            correlation, p_value = calculate_spearman_correlation(df_psma001_02_two_c_irre, df_psma001_02_two_c_re)
            print(f"Spearman Correlation: {correlation}")
            print(f"P-value: {p_value}")
            plot_correlation(df_psma001_02_two_c_irre, df_psma001_02_two_c_re, correlation, p_value)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("One or both of the specified Patient_IDs are missing.")
    if False:
        for patient_id, df in unique_dfs.items():
            df.to_csv(f"Patient_{patient_id}.csv", index=False)