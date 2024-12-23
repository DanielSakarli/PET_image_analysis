import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the required columns exist
    required_columns = ["Patient_ID", "Model", "K1", "k2", "k3", "k4", "vB"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")
    
    # Group the data by Patient_ID and Model
    grouped = df.groupby(["Patient_ID", "Model"])
    print(grouped.head())

    # Only take the first entry (lesion) of each unique Patient_ID where the Model is "2_Tissue_Compartments" and vB is not 0.05
    df = df[(df["Model"] == "2_Tissue_Compartments") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()
    print(df.head())

    # Split the Patient_ID into patient number and reconstruction method
    df[['Patient_Number', 'Reconstruction_Method']] = df['Patient_ID'].str.split('_', n=1, expand=True)
    print(df.head())

    # Create a pivot table for K1 values with Patient_Number as the index and Reconstruction_Method as the columns
    pivot_df = df.pivot(index="Patient_Number", columns="Reconstruction_Method", values="K1")
    
    print(f"Columns: {pivot_df.columns}")
    print(f"Rows: {pivot_df.shape[0]}")
    
    # Compute Spearman correlation for each patient
    correlation_results = {}
    for patient in pivot_df.index:
        patient_data = pivot_df.loc[patient].dropna()
        if len(patient_data) > 1:  # Ensure there are at least two reconstruction methods to correlate
            correlation = patient_data.corr(method="spearman")
            correlation_results[patient] = correlation
        else:
            correlation_results[patient] = None

    # Compute correlation matrix
    #correlation_matrix = pivot_df.corr(method="spearman")  # Use Pearson correlation
    print(correlation_results)

    
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
    file_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//PSMA001_k_values.csv"
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