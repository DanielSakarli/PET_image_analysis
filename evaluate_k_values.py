import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the required columns exist
    required_columns = ["Patient_ID", "Model", "K1", "k2", "k3", "k4", "vB"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Filter the data to include only relevant rows
    df_2tc_re = df[(df["Model"] == "2_Tissue_Compartments") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()
    # Filter the data to include only relevant rows
    df_2tc_irre = df[(df["Model"] == "2_Tissue_Compartments,_FDG") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()

    # Split the Patient_ID into Patient_Number and Reconstruction_Method
    df_2tc_re[['Patient_Number', 'Reconstruction_Method']] = df_2tc_re['Patient_ID'].str.split('_', n=1, expand=True)
    df_2tc_irre[['Patient_Number', 'Reconstruction_Method']] = df_2tc_irre['Patient_ID'].str.split('_', n=1, expand=True)

    # Calculate Spearman correlation for each k variable
    k_variables = ["K1", "k2", "k3", "k4"]
    correlation_results = {}
    for df, model in zip([df_2tc_re, df_2tc_irre], ["2_Tissue_Compartments", "2_Tissue_Compartments,_FDG"]):
        for var in k_variables:
            correlation_matrix = calculate_spearman_correlation(df, var, model)
            correlation_results[(model, var)] = correlation_matrix

    # Generate boxplots for all patients
    print(f"dataframe before sending to boxplot: \n{df_2tc_re}")
    create_boxplot(df_2tc_re, k_variables)

    return correlation_results
    

    
def calculate_spearman_correlation(df, k, model):
    """
    Calculates the Spearman correlation coefficient and p-value between k values.
    df: DataFrame containing the k values for each patient and reconstruction method.
    k: The k variable to calculate the correlation for (either K1, k2, k3, or k4).
    model: The compartment model type.
    """
    # Create a pivot table for k values across all patients
    pivot_df = df.pivot(index="Patient_Number", columns="Reconstruction_Method", values=k)
    print(f"\nPivot Table for {k} and {model}:\n", pivot_df)

    # Compute the Spearman correlation for reconstruction methods globally
    global_correlation_matrix = pivot_df.corr(method="spearman")
    print(f"Global Spearman Correlation Matrix for {k} and {model}:\n", global_correlation_matrix)

    return global_correlation_matrix


def create_boxplot(df, k_variables):
    """
    Create boxplots for the specified k variables (K1, k2, k3, k4) for a single patient.
    The boxplots for all k variables are drawn in the same figure.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        k_variables (list): List of variables to create boxplots for (e.g., ["K1", "k2", "k3", "k4"]).
    """
    # Prepare data for plotting
    data_for_plotting = []

    for k_variable in k_variables:
        for patient in df["Patient_Number"].unique():
            # Filter the data for the specific patient
            patient_data = df[df["Patient_Number"] == patient]

            if patient_data.empty:
                continue

            # Extract k_variable values for the patient
            values = patient_data[k_variable]

            # Calculate the IQR and identify outlier thresholds
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out outliers
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

            # Append to the plotting data
            data_for_plotting.append(
                pd.DataFrame({
                    "Values": filtered_values,
                    "Patient": patient,
                    "Variable": k_variable
                })
            )

    # Combine all patient data into one DataFrame
    combined_data = pd.concat(data_for_plotting)

    # Create the boxplot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=combined_data,
        x="Variable",
        y="Values",
        hue="Patient",
        dodge=True,  # Overlay boxplots for each variable
        linewidth=1.5
    )
    sns.despine()  # Remove top and right axes
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for better readability
    plt.title("Boxplots of k Variables for All Reconstruction Settings", fontsize=14)
    plt.xlabel("k Variables", fontsize=12)
    plt.ylabel("k Values [min⁻¹]", fontsize=12)
    plt.legend(title="Patient", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if False:
        # Prepare data for plotting
        boxplot_data = {k_variable: [] for k_variable in k_variables}
        labels = []

        for patient in df["Patient_Number"].unique():
            # Filter the data for the specific patient
            patient_data = df[df["Patient_Number"] == patient]

            if patient_data.empty:
                continue

            for k_variable in k_variables:
                # Extract k_variable values for the patient
                values = patient_data[k_variable]
                print(f"Values for patient {patient}: {values}")
                print(f"Type of values: {type(values)}")
                
                # Calculate the IQR and identify outlier thresholds
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter out outliers (ensure values is a Series)
                filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

                # Append data
                boxplot_data[k_variable].append(filtered_values.tolist())  # Convert to list

            # Append patient label
            labels.append(patient)

        # Plot all boxplots in the same figure
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab20.colors  # Use a colormap with enough colors for all patients
        positions = []
        for i, k_variable in enumerate(k_variables):
            for j, patient in enumerate(labels):
                positions.append(i * (len(labels) + 1) + j + 1)
                ax.boxplot(boxplot_data[k_variable][j], positions=[positions[-1]], patch_artist=True,
                        boxprops=dict(facecolor=colors[j % len(colors)], color=colors[j % len(colors)]),
                        medianprops=dict(color='black'))

        ax.set_xticks([i * (len(labels) + 1) + (len(labels) + 1) / 2 for i in range(len(k_variables))])
        ax.set_xticklabels(k_variables)
        ax.set_xlabel("k Variables")
        ax.set_ylabel("Values")
        ax.set_title("Value Distribution Across Patients and Variables")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//All_patients_k_values.csv"
    process_csv(file_path)
    
    if False:
        # Get DataFrames for specific Patient_ID and Model
        df_psma001_02_two_c_irre = unique_dfs.get(("PSMA001_02", "2_Tissue_Compartments"))    # two comparments irreversible
        df_psma001_02_two_c_re = unique_dfs.get(("PSMA001_02", "2_Tissue_Compartments,_FDG"))      # two compartments reversible

        if df_psma001_02_two_c_irre is not None and df_psma001_02_two_c_re is not None:
            # Calculate Spearman correlation
            try:
                correlation, p_value = calculate_spearman_correlation(df_psma001_02_two_c_irre, df_psma001_02_two_c_re)
                print(f"Spearman Correlation: {correlation}")
                print(f"P-value: {p_value}")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("One or both of the specified Patient_IDs are missing.")