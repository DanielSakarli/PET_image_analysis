import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def process_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the required columns exist
    required_columns = ["Patient_ID", "Model", "K1", "k2", "k3", "k4", "vB"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")
    
    # Filter the data to include only relevant rows
    df = df[(df["Model"] == "2_Tissue_Compartments") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()

    # Split the Patient_ID into Patient_Number and Reconstruction_Method
    df[['Patient_Number', 'Reconstruction_Method']] = df['Patient_ID'].str.split('_', n=1, expand=True)

    # Create a pivot table for K1 values
    pivot_df = df.pivot(index="Patient_Number", columns="Reconstruction_Method", values="K1")
    print("pivot:", pivot_df)
    # Print the structure of the pivot table
    print(f"Columns: {pivot_df.columns}")
    print(f"Rows: {pivot_df.shape[0]}")

    # Compute Spearman correlation for each patient
    correlation_results = {}
    for patient in pivot_df.index:
        # Get all reconstruction K1 values for this patient as a DataFrame
        #patient_data = pivot_df.loc[[patient]].T.dropna()  # Transpose and drop rows with NaN values
        # Convert Series to DataFrame
        
        patient_data = pivot_df.loc[patient].dropna()  # Drop NaN values
        
        print("patient_data:", patient_data)
        #patient_data.columns = [patient]  # Set the column name to the patient ID

        if len(patient_data) > 1:  # Ensure at least two reconstruction methods
            patient_data_df = patient_data.to_frame().T
            # Compute the pairwise Spearman correlation for reconstruction methods
            correlation_matrix = patient_data_df.corr(method="spearman")
            correlation_results[patient] = correlation_matrix
        else:
            correlation_results[patient] = None  # Not enough data to compute correlation

    # Print the correlation results
    for patient, correlation_matrix in correlation_results.items():
        if correlation_matrix is not None:
            print(f"Patient {patient}: Correlation Matrix:\n{correlation_matrix}")
        else:
            print(f"Patient {patient}: Not enough data for correlation.")

    
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