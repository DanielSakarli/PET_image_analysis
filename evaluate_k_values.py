import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tkinter import messagebox
from scipy.stats import wilcoxon
import itertools
import numpy as np
from scipy.stats import linregress

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
    #create_boxplot(df_2tc_re, k_variables)
    #create_lineplots(df_2tc_re, k_variables)
    create_scatterplot(df_2tc_re, k_variables)
    # With the irreversible model, exclude the k4 variable, so it doesn't appear as a x label in the plot
    k_variables.remove("k4")
    #create_boxplot(df_2tc_irre, k_variables)
    #create_lineplots(df_2tc_irre, k_variables)
    create_scatterplot(df_2tc_irre, k_variables)
    
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

def create_scatterplot(df, k_variables):
    """
    Create scatter plots for the specified k variables (K1, k2, k3, k4) for all patients.
    
    df: DataFrame containing the k values for each patient and reconstruction method.
    k: The k variable to calculate the correlation for (either K1, k2, k3, or k4).
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
        # Convert to numeric and drop rows with NaN in k_variable
        df[k_variable] = pd.to_numeric(df[k_variable], errors="coerce")
        df = df.dropna(subset=[k_variable])
        grouped = df.groupby("Reconstruction_Method")[k_variable]
        mean_ = grouped.mean()
        std_ = grouped.std()
        count_ = grouped.count()

        sem_ = std_ / np.sqrt(count_)
        ci95_ = 1.96 * sem_

        # Print the mean and 95% CI for each recon
        print(f"Mean {k_variable} by Reconstruction (± 95% CI):")
        print(mean_)
        print(ci95_)

        # Optionally, create a bar plot
        plt.figure(figsize=(6, 4))
        plt.bar(mean_.index, mean_.values, yerr=ci95_.values, capsize=5)
        plt.title(f"Mean {k_variable} by Reconstruction (± 95% CI)")
        plt.xticks(rotation=45)
        plt.ylabel(k_variable)
        plt.tight_layout()
        plt.show()

        # Pivot
        pivot_df = df.pivot(index="Patient_Number", 
                            columns="Reconstruction_Method", 
                            values=k_variable)

        # Ensure column labels are strings (e.g. "04", "05")
        pivot_df.columns = pivot_df.columns.astype(str)

        # Drop patient_numbers PSMA001 and PSMA006
        #pivot_df = pivot_df.drop(["PSMA001", "PSMA006"], errors="ignore")

        # Check that both "04" and "05" exist
        #if "04" not in pivot_df.columns or "05" not in pivot_df.columns:
        #    print("Error: Reconstruction Method '04' or '05' not found in pivot table. Skipping this k_variable.")
        #    continue

        # Drop rows where either "04" or "05" is NaN
        pivot_df = pivot_df.dropna()

        print("k_variable:", k_variable)
        # Discard the outlier in column '04_b' for row 'PSMA002'
        if k_variable == "Flux" and "04_b" in pivot_df.columns and "PSMA002" in pivot_df.index:
            pivot_df.loc["PSMA002", "04_b"] = np.nan

        # If nothing left after dropping, skip
        if pivot_df.empty:
            print(f"No valid rows for {k_variable} after dropping NaNs.")
            continue
        # We'll store the p-values here
        pvals = []

        # Loop over every column except "04"
        for col in pivot_df.columns:
            if col == "04":
                continue  # skip comparing '04' with itself
            
            # Drop rows where y-column is NaN
            df_pair = pivot_df.dropna(subset=[col])
            if df_pair.empty:
                print(f"No valid data for column '{col}'. Skipping.")
                continue

            # x is always "04"
            x = df_pair["04"]
            # y is the current column
            y = df_pair[col]

            # Perform linear regression
            result = linregress(x, y)

            slope = result.slope
            intercept = result.intercept
            r_value = result.rvalue           # Correlation coefficient (r)
            r_squared = result.rvalue**2      # R-squared
            p_value = result.pvalue           # p-value for the slope
            std_err = result.stderr           # Standard error of the slope

            # Print them out or log them
            print("-------------------------------------")
            print(f"Comparing '04' vs '{col}' for {k_variable}")
            print(f"  Slope:        {slope}")
            print(f"  Intercept:    {intercept}")
            print(f"  R-squared:    {r_squared}")
            print(f"  p-value:      {p_value}")
            print(f"  std. error:   {std_err}")

            # Append p-value to our list
            pvals.append(p_value)
            # Scatter plot using Seaborn
            sns.scatterplot(x=x, y=y, label=f"04 vs {col}. R²={r_squared:.3f}")

            # Linear regression line using Seaborn
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            sns.lineplot(x=x_fit, y=y_fit, color="red")

            # Scatter plot
            #plt.scatter(x, y, alpha=0.7, label=f"04vs{col}. R²={r_squared:.3f}")
            
            # Linear regression line using NumPy polyfit
            #slope, intercept = np.polyfit(x, y, 1)
            #x_fit = np.linspace(x.min(), x.max(), 100)
            #y_fit = slope * x_fit + intercept
            #plt.plot(x_fit, y_fit, color="red")

            # Determine axis limits automatically (or set your own)
            if k_variable == "Flux":
                min_val = 0.01
                max_val = 0.06
            elif k_variable == "K1":
                min_val = 0.04
                max_val = 0.16
            elif k_variable == "k2":
                min_val = 0.01
                max_val = 0.16
            elif k_variable == "k3":
                min_val = 0.01
                max_val = 0.16
            elif k_variable == "k4":
                min_val = 0
                max_val = 0.05
                
            # Plot 45° reference line
            plt.plot([min_val, max_val], [min_val, max_val],
                    linestyle="--", color="gray")

            # Set axes
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            if k_variable == "Flux":
                plt.xlabel(f"$K_i$ [min$^{{-1}}$] (Reconstruction 04)", fontsize=12)
                plt.ylabel(f"$K_i$ [min$^{{-1}}$] (All other reconstructions)", fontsize=12)
            else:
                plt.xlabel(f"{k_variable} [min$^{{-1}}$] (Reconstruction 04)", fontsize=12)
                plt.ylabel(f"{k_variable} [min$^{{-1}}$] (All other reconstructions)", fontsize=12)
            if k_variable == "Flux":
                plt.title(f"{model_name} – {region_name} – $K_i$", fontsize=14)
            else:
                plt.title(f"{model_name} – {region_name} – {k_variable}", fontsize=14)
            plt.legend()
            sns.despine()
            #plt.grid(True)
            #plt.tight_layout()
        # After we finish looping all columns, compute the mean p-value (if pvals is not empty).
        if pvals:
            mean_p = np.mean(pvals)
            print(f"Mean p-value for {k_variable} across all recon comparisons: {mean_p:.6g}")
        else:
            print(f"No p-values found for {k_variable}.")
        # Show the plot to the user
        plt.show(block=False)
        save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//Scatterplots//AIF"
        png_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_04_vs_all_recon_settings.png')
        pdf_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_04_vs_all_recon_settings.pdf')
        pickle_path = os.path.join(save_path, f'{k_variable}_AIF_{model_name}_for_{region_name}_04_vs_all_recon_settings.pickle')
        
        answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here:\n{save_path}\nas\n{png_path}?")
        if answer:
            # Save the plot as PNG, PDF, and pickle files
            plt.savefig(png_path)
            plt.savefig(pdf_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(plt.gcf(), f)

        # Show the plot again to ensure it remains visible after saving it
        plt.show()

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

    # Store all p-values
    p_values_list = []

    for k_variable in k_variables: 
        data_for_plotting = []

        for recon_setting in df["Reconstruction_Method"].unique():
            # Filter data for the specific reconstruction method
            recon_data = df[df["Reconstruction_Method"] == recon_setting]

            if recon_data.empty:
                continue

            # Convert k_variable to numeric and drop NaN values
            values = pd.to_numeric(recon_data[k_variable], errors="coerce").dropna()

            # Calculate IQR and filter outliers
            Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

            # Append to the dataset
            data_for_plotting.append(
                pd.DataFrame({
                    "Values": filtered_values,
                    "Recon_Setting": recon_setting,
                    "Patient Number": recon_data["Patient_Number"]
                })
            )

        # Combine data
        combined_data = pd.concat(data_for_plotting)

        # Perform Wilcoxon test for each pair of reconstruction methods
        recon_methods = combined_data["Recon_Setting"].unique()

        for recon1, recon2 in itertools.combinations(recon_methods, 2):
            # Get paired samples per patient
            paired_data = combined_data.pivot(index="Patient Number", columns="Recon_Setting", values="Values")

            if recon1 in paired_data.columns and recon2 in paired_data.columns:
                paired_samples = paired_data[[recon1, recon2]].dropna()  # Drop patients missing either value

                if len(paired_samples) > 1:  # Wilcoxon test requires at least two paired samples
                    stat, p = wilcoxon(paired_samples[recon1], paired_samples[recon2])
                    p_values_list.append({"k_variable": k_variable, "Recon1": recon1, "Recon2": recon2, "p-value": p})

    # Convert p-values list to a DataFrame
    p_values_df = pd.DataFrame(p_values_list)

    # Print results in a formatted table
    if not p_values_df.empty:
        print("\nWilcoxon Signed-Rank Test Results:")
        print(p_values_df.to_string(index=False))
    else:
        print("\nNo valid paired samples available for Wilcoxon test.")

    if False:    
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
    