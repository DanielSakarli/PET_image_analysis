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
import scipy.stats as stats

def process_csv(file_path):
    flag_calculate_wilcoxon_test = False
    flag_do_scatterplot = True
    flag_do_boxplot = False

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=";")
    
    # Ensure the required columns exist
    required_columns = ["Patient_ID", "Model", "K1", "k2", "k3", "k4", "vB", "Flux"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")
    
    # Discard rows where vB is 0 or 0.05
    df = df[(df["vB"] != 0) & (df["vB"] != 0.05)]

    # Filter the data to include only relevant rows
    df_2tc_re = df[(df["Model"] == "2_Tissue_Compartments") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()
    df_2tc_irre = df[(df["Model"] == "2_Tissue_Compartments,_FDG") & (df["vB"] != 0.05)].groupby("Patient_ID").first().reset_index()

    # Split the Patient_ID into Patient_Number and Reconstruction_Method
    df_2tc_re[['Patient_Number', 'Reconstruction_Method']] = df_2tc_re['Patient_ID'].str.split('_', n=1, expand=True)
    df_2tc_irre[['Patient_Number', 'Reconstruction_Method']] = df_2tc_irre['Patient_ID'].str.split('_', n=1, expand=True)

    
    k_variables_re = ["Flux", "K1", "k2", "k3", "k4"]
    k_variables_irre = ["Flux", "K1", "k2", "k3"]

    if flag_calculate_wilcoxon_test:
        # Perform Wilcoxon signed rank test for each k variable
        p_matrix_re = []
        p_matrix_irre = []
        for k_variable in k_variables_re:
            p_matrix_re.append(wilcoxon_test(df_2tc_re, k_variable))
            if k_variable in k_variables_irre:
                p_matrix_irre.append(wilcoxon_test(df_2tc_irre, k_variable))

        

        print("\nWilcoxon Signed-Rank Test Results for Irreversible Model:")
        for k_variable, p_matrix in zip(k_variables_irre, p_matrix_irre):
            print(f"\n{k_variable}:\n{p_matrix}")
        
        print("\nWilcoxon Signed-Rank Test Results for Reversible Model:")
        for k_variable, p_matrix in zip(k_variables_re, p_matrix_re):
            print(f"\n{k_variable}:\n{p_matrix}")

    if False:
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
    
    if flag_do_boxplot:
        # Filter the data but keep ALL rows (no groupby/first)
        df_2tc_re = df[
            (df["Model"] == "2_Tissue_Compartments") & 
            (df["vB"] != 0.05)
        ].reset_index(drop=True)

        df_2tc_irre = df[
            (df["Model"] == "2_Tissue_Compartments,_FDG") & 
            (df["vB"] != 0.05)
        ].reset_index(drop=True)
        # Split the Patient_ID into Patient_Number and Reconstruction_Method
        df_2tc_re[['Patient_Number', 'Reconstruction_Method']] = df_2tc_re['Patient_ID'].str.split('_', n=1, expand=True)
        df_2tc_irre[['Patient_Number', 'Reconstruction_Method']] = df_2tc_irre['Patient_ID'].str.split('_', n=1, expand=True)

        create_boxplot(df_2tc_irre, k_variables_irre)
        create_boxplot(df_2tc_re, k_variables_re)
        #wilcoxon_for_PMOD_stability(df_2tc_re, df_2tc_irre, k_variables_irre)

    if flag_do_scatterplot:
        create_scatterplot(df_2tc_irre, k_variables_irre)
        create_scatterplot(df_2tc_re, k_variables_re)
    #return correlation_results

def wilcoxon_for_PMOD_stability(df_2tc_re, df_2tc_irre, k_variables):
    """
    Compare the given k_variables between df_2tc_re and df_2tc_irre
    using the Wilcoxon signed-rank test, matching on Patient_ID.
    """
    print(df_2tc_irre.head())  
    print(df_2tc_re.head())

    # For each k-variable, run Wilcoxon signed-rank
    results = {}
    for k in k_variables:
        # Extract the corresponding columns
        re_vals = pd.to_numeric(df_2tc_re[f"{k}"], errors="coerce").dropna()
        irre_vals = pd.to_numeric(df_2tc_irre[f"{k}"], errors="coerce").dropna()
       
        stat, p_val = stats.wilcoxon(re_vals, irre_vals)
        results[k] = (stat, p_val)
        print(f"{k}: Wilcoxon test statistic = {stat}, p-value = {p_val}")
    

    return results

def wilcoxon_test(df, k_variable):
    """
    Perform Wilcoxon signed-rank tests for a given k_variable across all reconstruction methods.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'Patient_Number', 'Reconstruction_Method', and k_variable.
        k_variable (str): The k-variable to analyze (e.g., 'K1', 'k2', etc.).
    
    Returns:
        pd.DataFrame: A matrix of p-values comparing reconstruction methods.
    """
    # Make sure that the k_variable values are numeric and not strings
    df[k_variable] = pd.to_numeric(df[k_variable], errors="coerce")
    # Exclude all values above 2 (outlier detetction)
    df = df[df[k_variable] <= 2]

    # Get unique reconstruction methods
    recon_methods = df["Reconstruction_Method"].unique()
    
    # Initialize an empty DataFrame for storing p-values
    p_matrix = pd.DataFrame(index=recon_methods, columns=recon_methods)
    
    # Perform Wilcoxon signed-rank test for each pair of reconstruction methods
    for method1, method2 in itertools.combinations(recon_methods, 2):
        # Extract the k-variable values for both methods, matching by Patient_Number
        df1 = df[df["Reconstruction_Method"] == method1][["Patient_Number", k_variable]]
        df2 = df[df["Reconstruction_Method"] == method2][["Patient_Number", k_variable]]
        print("method 1", method1)
        print("method 2", method2)
        # Merge to ensure we compare the same patients
        merged = pd.merge(df1, df2, on="Patient_Number", suffixes=("_1", "_2"))
        print("merged dataframe:\n", merged)
        if not merged.empty:
            # Perform Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(merged[f"{k_variable}_1"], merged[f"{k_variable}_2"])
            p_value = round(p_value, 3)  # Round to three decimal places
            p_matrix.loc[method1, method2] = p_value
            print("p-value:", p_value)
            p_matrix.loc[method2, method1] = p_value  # Mirror value
        else:
            p_matrix.loc[method1, method2] = None
            p_matrix.loc[method2, method1] = None
    
    # Fill diagonal with NaN (comparison with itself is not needed)
    for method in recon_methods:
        p_matrix.loc[method, method] = None
    
    return p_matrix
    
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

        ##############################################################
        ##### Analyze if the residuals and the k values themselves are
        # normally distributed. Necessary to do linear regression.
        ##############################################################
        # If the data comes as a DataFrame
        
        
        all_values = []
        differences = []

        ref_values = grouped.get_group('04').values  # Reference values from '04'

        for name, group in grouped:
            
            filtered_values = [value for value in group.values if value <= 2]
            all_values.extend(filtered_values)

            # Calculate differences
            for i in range(min(len(ref_values), len(filtered_values))):
                differences.append(ref_values[i] - filtered_values[i])

        all_values = np.array(all_values)
        differences = np.array(differences)

        # Plotting combined histogram with KDE
        plt.figure(figsize=(10, 6))
        sns.histplot(all_values, kde=True, bins=100, color='lightcoral')
        plt.title('Combined Histogram of k values across all Reconstruction Methods')
        plt.xlabel('k values')
        plt.ylabel('Frequency')
        plt.show()

        # Plotting histogram of differences
        plt.figure(figsize=(10, 6))
        sns.histplot(differences, kde=True, bins=100, color='steelblue')
        plt.title('Histogram of Differences (Reference: 04)')
        plt.xlabel('Difference (04 - Other Methods)')
        plt.ylabel('Frequency')
        plt.show()
        # Print out the reconstruction methods and patient_id of the values in filtered_values which are above 0.03
        print("Reconstruction methods of values above 0.03:")
        print(df[df[k_variable] > 0.03]["Reconstruction_Method"].unique())
        print("Patient IDs of values above 0.03:")
        print(df[df[k_variable] > 0.03]["Patient_ID"].unique())

        # Print the mean and 95% CI for each recon
        print("All k-variable values: \n", grouped.apply(list))
        print(f"Mean {k_variable} by Reconstruction (± 95% CI):")
        print("Mean values: \n", mean_)
        print("CI95 values: \n", ci95_)
        if False:
            # Optionally, create a bar plot
            plt.figure(figsize=(6, 4))
            plt.bar(mean_.index, mean_.values, yerr=ci95_.values, capsize=5)
            plt.title(f"Mean {k_variable} by Reconstruction (± 95% CI)")
            plt.xticks(rotation=45)
            plt.ylabel(k_variable)
            plt.tight_layout()
            plt.show()

        
        # Include only reconstruction_method 04 and 05 for test purposes
        #df = df[df["Reconstruction_Method"].isin(["04", "02_a"])]
        
        # Pivot
        # Filter out k values > 2
        df = df[df[k_variable] <= 2]
        pivot_df = df.pivot(index="Patient_Number", 
                            columns="Reconstruction_Method", 
                            values=k_variable)
        print("Pivot dataframe:\n", pivot_df)

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
        #if k_variable == "Flux" and "04_b" in pivot_df.columns and "PSMA002" in pivot_df.index:
        #    pivot_df.loc["PSMA002", "04_b"] = np.nan

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
            print(f"  Corr. coeff.:      {r_value}")
            print(f"  R-squared:    {r_squared}")
            print(f"  p-value:      {p_value}")
            print(f"  std. error:   {std_err}")

            # Append p-value to our list
            pvals.append(p_value)
            # Scatter plot using Seaborn
            # Remove the leading 0 in col string and check if col ends on _a or _b, if so remove the _a and _b
            
            filter_size = "3mm"
            col_temp = col.lstrip("0")
            if col.endswith("_a"):
                col_temp = col_temp.rstrip("_a")
                filter_size = "5mm"
            elif col.endswith("_b"):
                col_temp = col_temp.rstrip("_b")
                filter_size = "7mm"

            if intercept<0:
                legend_entry = col_temp + f"i, {filter_size}: y={slope:.3f}x{intercept:.3f}"
            else:
                legend_entry = col_temp + f"i, {filter_size}: y={slope:.3f}x+{intercept:.3f}"

            sns.scatterplot(x=x, y=y, label=legend_entry) #label=f"04 vs {col}. R²={r_squared:.3f}"

            # Linear regression line using Seaborn
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            sns.lineplot(x=x_fit, y=y_fit) #, color="red"

            # Scatter plot
            #plt.scatter(x, y, alpha=0.7, label=f"04vs{col}. R²={r_squared:.3f}")
            
            # Linear regression line using NumPy polyfit
            #slope, intercept = np.polyfit(x, y, 1)
            #x_fit = np.linspace(x.min(), x.max(), 100)
            #y_fit = slope * x_fit + intercept
            #plt.plot(x_fit, y_fit, color="red")

            # Determine axis limits automatically (or set your own)
            if k_variable == "Flux":
                min_val = 0.005
                max_val = 0.06
            elif k_variable == "K1":
                min_val = 0.003
                max_val = 0.16
            elif k_variable == "k2":
                min_val = 0.02
                max_val = 0.16
            elif k_variable == "k3":
                min_val = 0.01
                max_val = 0.05
            elif k_variable == "k4":
                min_val = 0
                max_val = 0.05
                
            # Plot 45° reference line
            plt.plot([min_val, max_val], [min_val, max_val],
                    linestyle="--", color="gray")
            # Model name for title
            if model_name == "Two Compartment Reversible Model":
                model = "Reversible 2TCM-4k"
            elif model_name == "Two Compartment Irreversible Model":
                model = "Irreversible 2TCM-3k"

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
                plt.title(f"{model} – {region_name} – $K_i$", fontsize=14)
            else:
                plt.title(f"{model} – {region_name} – {k_variable}", fontsize=14)
            plt.legend(fontsize=8)
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
    Plots all entries for each k variable together in one boxplot figure.
    For example, if the patient_id is 'PSMA007_04', you get one box for K1,
    one for k2, etc., side by side.
    """
    patient_id = "PSMA007_04"
    # Filter for the chosen patient
    df_patient = df[df['Patient_ID'] == patient_id].copy()

    # Convert each k-variable column into a "long" format
    df_melted = df_patient.melt(
        id_vars=["Patient_ID", "Reconstruction_Method"],
        value_vars=k_variables,
        var_name="k_variable",
        value_name="value"
    )

    # Drop rows where value is NaN
    df_melted.dropna(subset=["value"], inplace=True)
    
    if df["Model"].iloc[0] == "2_Tissue_Compartments":
        model_name = "Two Compartment Reversible Model"
    elif df["Model"].iloc[0] == "2_Tissue_Compartments,_FDG":
        model_name = "Two Compartment Irreversible Model"
    else:
        model_name = "Unknown Model"

    # Plot all k_variable boxplots in the same figure
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="k_variable", y="value", data=df_melted)
    sns.stripplot(x="k_variable", y="value", data=df_melted, color='black', alpha=0.5)

    plt.title(f"{model_name} Stability Test")
    plt.xlabel("k_variable")
    plt.ylabel("Value [min$^{-1}$]")
    plt.tight_layout()
    plt.show()

    # Compute summary stats for each k_variable group
    print(f"Data used for statistics: ", df_melted.groupby("k_variable")["value"].describe())
    group_stats = df_melted.groupby("k_variable")["value"].agg(["mean", "var", "std", "count"])
    group_stats["sem"] = group_stats["std"] / np.sqrt(group_stats["count"])
    # 95% CI using ~1.96 multiplier for large samples
    group_stats["ci_95"] = 1.96 * group_stats["sem"]

    # Print the statistics
    print(f"\n{model_name}\nVariance and 95% CI by k_variable for patient:", patient_id)
    print(group_stats[["std", "mean", "var", "sem", "ci_95"]])

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
    file_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Kinetic Modelling//All_patients_k_values_AIF_with_inital_parameters_equal_0.csv" #stability_test.csv
    process_csv(file_path)
    