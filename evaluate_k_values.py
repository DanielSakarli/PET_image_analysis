import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tkinter import messagebox

def process_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the required columns exist
    required_columns = ["Patient_ID", "Model", "K1", "k2", "k3", "k4", "vB", "Flux"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Filter the data to include only relevant rows
    df_2tc_re = df[(df["Model"] == "2_Tissue_Compartments") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()
    df_2tc_irre = df[(df["Model"] == "2_Tissue_Compartments,_FDG") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()

    # Split the Patient_ID into Patient_Number and Reconstruction_Method
    df_2tc_re[['Patient_Number', 'Reconstruction_Method']] = df_2tc_re['Patient_ID'].str.split('_', n=1, expand=True)
    df_2tc_irre[['Patient_Number', 'Reconstruction_Method']] = df_2tc_irre['Patient_ID'].str.split('_', n=1, expand=True)
    
    if False:
        # Divide the df into the different regions lesion, healthy_prostate, and gluteus_maximus
        df_2tc_re_lesion = df_2tc_re[df_2tc_re["Region"] == "lesion"]
        df_2tc_re_healthy_prostate = df_2tc_re[df_2tc_re["Region"] == "healthy_prostate"]
        df_2tc_re_gluteus_maximus = df_2tc_re[df_2tc_re["Region"] == "gluteus_maximus"]
        df_2tc_irre_lesion = df_2tc_irre[df_2tc_irre["Region"] == "lesion"]
        df_2tc_irre_healthy_prostate = df_2tc_irre[df_2tc_irre["Region"] == "healthy_prostate"]
        df_2tc_irre_gluteus_maximus = df_2tc_irre[df_2tc_irre["Region"] == "gluteus_maximus"]

    # Calculate Spearman correlation for each k variable
    k_variables = ["Flux", "K1", "k2", "k3", "k4"]
    correlation_results = {}
    for df, model in zip([df_2tc_re, df_2tc_irre], ["2_Tissue_Compartments", "2_Tissue_Compartments,_FDG"]):
        for var in k_variables:
            correlation_matrix = calculate_spearman_correlation(df, var, model)
            correlation_results[(model, var)] = correlation_matrix
    print("I'm here")
    # Generate boxplots for all patients and k variables
    # With the reversible model
    create_boxplot(df_2tc_re, k_variables)
    #create_lineplots(df_2tc_re, k_variables)
    # With the irreversible model, exclude the k4 variable, so it doesn't appear as a x label in the plot
    k_variables.remove("k4")
    create_boxplot(df_2tc_irre, k_variables)
    #create_lineplots(df_2tc_irre, k_variables)
    
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
    
    # Get model and region name for the title
    if df["Model"].iloc[0] == "2_Tissue_Compartments":
        model_name = "Two Compartment Reversible Model"
    elif df["Model"].iloc[0] == "2_Tissue_Compartments,_FDG":
        model_name = "Two Compartment Irreversible Model"
    else:
        model_name = "Unknown Model"

    if df["Region"].iloc[0] == "lesion":
        region_name = "Lesion"
    elif df["Region"].iloc[0] == "healthy_prostate":
        region_name = "Healthy Prostate"
    elif df["Region"].iloc[0] == "gluteus_maximus":
        region_name = "Gluteus Maximus"

    
    for k_variable in k_variables:
        data_for_plotting = []
        for recon_setting in df["Reconstruction_Method"].unique():
                # Filter the data for the specific recon method
            recon_data = df[df["Reconstruction_Method"] == recon_setting]

            if recon_data.empty:
                continue

            # Convert k_variable to numeric
            values = pd.to_numeric(recon_data[k_variable], errors="coerce")
            
            # Drop NaN values
            values = values.dropna()

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
                    "Recon_Setting": recon_setting,
                    "Patient Number": recon_data["Patient_Number"]
                })
            )
        
        # Combine all patient data into one DataFrame
        combined_data = pd.concat(data_for_plotting)

        # Create the boxplot using Seaborn
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            data=combined_data,
            x="Recon_Setting",
            y="Values",
            dodge=True,  # Overlay boxplots for each variable
            linewidth=1.5
        )
        sns.stripplot(
            data=combined_data,
            x="Recon_Setting",
            y="Values",
            hue="Patient Number",
            palette="Set2",
            alpha=0.75,
            jitter=True,
            dodge=True
        )
        sns.despine()  # Remove top and right axes
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for better readability
        plt.title(f"{model_name} for {region_name} - {k_variable} with all Recon Settings", fontsize=14)
        plt.xlabel("Reconstruction Method", fontsize=12)
        plt.ylim(0, 0.2)
        plt.ylabel(f"{k_variable} Values [min⁻¹]", fontsize=12)
        plt.tight_layout()
        # Show the plot to the user
        plt.show(block=False)
        save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//Boxplots//AIF"
        png_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_all_recon_settings.png')
        pdf_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_all_recon_settings.pdf')
        pickle_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_all_recon_settings.pickle')
        
        answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here:\n{save_path}\nas\n{png_path}?")
        if answer:
            # Save the plot as PNG, PDF, and pickle files
            plt.savefig(png_path)
            plt.savefig(pdf_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(plt.gcf(), f)

        # Show the plot again to ensure it remains visible after saving it
        plt.show()
    if False:
        for k_variable in k_variables:
            for patient in df["Patient_Number"].unique():
                # Filter the data for the specific patient
                patient_data = df[df["Patient_Number"] == patient]

                if patient_data.empty:
                    continue

                # Convert k_variables to numeric
                values = pd.to_numeric(patient_data[k_variable], errors="coerce")
                
                # Drop NaN values
                values = values.dropna()

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
            hue="Recon_Setting",
            dodge=True,  # Overlay boxplots for each variable
            linewidth=1.5
        )
        sns.despine()  # Remove top and right axes
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for better readability
        plt.title(f"{model_name} for {region_name} with all Recon Settings", fontsize=14)
        plt.xlabel("k Variables", fontsize=12)
        plt.ylim(0, 0.5)
        plt.ylabel("k Values [min⁻¹]", fontsize=12)
        plt.legend(
                title="Patient",
                loc="upper right",  # Position within the plot
                bbox_to_anchor=(0.98, 0.98)  # Fine-tune the location
        )
        plt.tight_layout()
        # Show the plot to the user
        plt.show(block=False)

def create_lineplots(df, k_variables):
    """
    Create line plots for the specified k variables (K1, k2, k3, k4) with reconstruction methods
    on the x-axis and patient data as separate lines.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        k_variables (list): List of k variables to plot (e.g., ["K1", "k2", "k3", "k4"]).
    """

    # Get model and region name for the title
    if df["Model"].iloc[0] == "2_Tissue_Compartments":
        model_name = "Two Compartment Reversible Model"
    elif df["Model"].iloc[0] == "2_Tissue_Compartments,_FDG":
        model_name = "Two Compartment Irreversible Model"
    else:
        model_name = "Unknown Model"

    if df["Region"].iloc[0] == "lesion":
        region_name = "Lesion"
    elif df["Region"].iloc[0] == "healthy_prostate":
        region_name = "Healthy Prostate"
    elif df["Region"].iloc[0] == "gluteus_maximus":
        region_name = "Gluteus Maximus"

    for k_variable in k_variables:
        # Prepare data for plotting
        data_for_plotting = []

        for patient in df["Patient_Number"].unique():
            # Filter the data for the specific patient
            patient_data = df[df["Patient_Number"] == patient]

            if patient_data.empty:
                continue

            # Append data to the plotting dataset
            data_for_plotting.append(
                pd.DataFrame({
                    "Reconstruction_Method": patient_data["Reconstruction_Method"],
                    k_variable: pd.to_numeric(patient_data[k_variable], errors="coerce"),
                    "Patient": patient
                }).dropna()
            )

        # Combine all patient data into one DataFrame
        combined_data = pd.concat(data_for_plotting)
        print(f"\nCombined Data for {k_variable}:\n", combined_data.head())
        # Create the line plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=combined_data,
            x="Reconstruction_Method",
            y=k_variable,
            hue="Patient",
            marker="o"
        )

        sns.despine()  # Remove top and right axes
        plt.title(f"{model_name} for {region_name} Across Reconstruction Methods", fontsize=14)
        plt.xlabel("Reconstruction Method", fontsize=12)
        plt.ylabel(f"{k_variable} Values [min$^{-1}$]", fontsize=12)
        #plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.legend(
            title="Patient",
            loc="upper right",  # Position within the plot
            bbox_to_anchor=(0.98, 0.98)  # Fine-tune the location
        )
        plt.tight_layout()
        # Show the plot to the user
        plt.show(block=False)
        if False:
            if k_variable == "Flux":
                plt.ylim(0, 0.1)
            if k_variable == "K1":
                plt.ylim(0, 0.5)
            if k_variable == "k2":
                plt.ylim(0, 0.5)
            if k_variable == "k3":
                plt.ylim(0, 0.2)
            if k_variable == "k4":
                plt.ylim(0, 0.05)
        save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//Lineplots//AIF"
        png_path = os.path.join(save_path, f'AIF_{model_name}_for_{region_name}_and_{k_variable}_variable_all_recon_settings.png')
        pdf_path = os.path.join(save_path, f'AIF_{model_name}_for_{region_name}_and_{k_variable}_variable_all_recon_settings.pdf')
        pickle_path = os.path.join(save_path, f'AIF_{model_name}_for_{region_name}_and_{k_variable}_variable_all_recon_settings.pickle')
        
        answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here:\n{save_path}\nas\n{png_path}?")
        if answer:
            # Save the plot as PNG, PDF, and pickle files
            plt.savefig(png_path)
            plt.savefig(pdf_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(plt.gcf(), f)

        # Show the plot again to ensure it remains visible after saving it
        plt.show()


if __name__ == "__main__":
    file_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//All_patients_k_values_AIF_with_inital_parameters_equal_0.csv"
    process_csv(file_path)
    