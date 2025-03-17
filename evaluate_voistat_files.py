import os
import pandas as pd
import scipy.stats as stats
from itertools import combinations
import numpy as np

# Sample data (replace this with your actual data)
data = {
    'Reconstruction Setting': [
        '2i, 3mm', '2i, 5mm', '2i, 7mm', '3i, 3mm', '3i, 5mm', '3i, 7mm',
        '4i, 3mm', '4i, 5mm', '4i, 7mm', '5i, 3mm', '5i, 5mm', '5i, 7mm'
    ],
    'PSMA001': [4.172, 5.078, 6.182, 4.244, 5.224, 6.338, 4.270, 5.306, 6.417, 4.271, 6.460, 5.356],
    'PSMA002': [4.847, 5.573, 6.137, 4.814, 5.640, 6.230, 4.742, 5.645, 6.244, 4.667, 5.640, 6.244],
    'PSMA005': [4.744, 5.728, 6.524, 4.182, 5.363, 6.321, 3.833, 5.126, 6.197, 3.588, 4.952, 6.107],
    'PSMA006': [4.836, 5.499, 6.047, 4.505, 5.284, 5.914, 4.249, 5.092, 5.775, 4.066, 4.951, 5.669],
    'PSMA007': [4.770, 5.326, 5.809, 4.925, 5.678, 6.153, 4.881, 5.849, 6.358, 4.763, 5.926, 6.482],
    'PSMA009': [2.974, 3.568, 4.188, 2.916, 3.580, 4.236, 2.863, 3.581, 4.260, 2.811, 3.576, 4.271],
    'PSMA010': [2.781, 3.305, 3.931, 2.721, 3.273, 3.919, 2.666, 3.234, 3.888, 2.614, 3.194, 3.850]
}

df = pd.DataFrame(data)


def compare_all_recon_settings():
    """
    Compare all reconstruction settings with each other using Wilcoxon signed-rank test.
    Print p-values as a matrix.
    """
    recon_settings = df['Reconstruction Setting'].tolist()
    num_settings = len(recon_settings)

    # Initialize a matrix to store p-values
    p_value_matrix = np.zeros((num_settings, num_settings))

    for i in range(num_settings):
        for j in range(num_settings):
            if i != j:
                data1 = df.loc[df['Reconstruction Setting'] == recon_settings[i]].iloc[:, 1:].values.flatten()
                data2 = df.loc[df['Reconstruction Setting'] == recon_settings[j]].iloc[:, 1:].values.flatten()
                _, p_value = stats.wilcoxon(data1, data2)
                p_value_matrix[i, j] = p_value
            else:
                p_value_matrix[i, j] = np.nan  # Leave diagonal as NaN

    # Convert to DataFrame for better readability
    p_value_df = pd.DataFrame(p_value_matrix, index=recon_settings, columns=recon_settings)

    print("P-value Matrix:\n")
    print(p_value_df)

# Run the comparison
compare_all_recon_settings()




def process_voistat_file(file_path):
    recon_id = os.path.splitext(os.path.basename(file_path))[0].split('_', 1)[1]
    header = None
    data = []

    # Read lines, identify header (line starting with "//"), skip comment lines ("#"), collect data
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('//'):
                header = line.strip('//').strip().split('\t')
                continue
            if line.startswith('#'):
                continue
            data.append(line.strip().split('\t'))
    
    # Create a DataFrame only if we found a header
    if not header:
        return recon_id, None, None, None  # No header found, return empty

    df = pd.DataFrame(data, columns=header)

    # Columns of interest
    columns_of_interest = [
        "VoiName(Region) [string]", 
        "Averaged [SUV]", 
        "Sd [SUV]"
    ]
    df_filtered = df[columns_of_interest].copy()

    # Convert to numeric
    df_filtered["Averaged [SUV]"] = pd.to_numeric(df_filtered["Averaged [SUV]"], errors="coerce")
    df_filtered["Sd [SUV]"] = pd.to_numeric(df_filtered["Sd [SUV]"], errors="coerce")

    # Filter rows where "VoiName(Region) [string]" == "lesion"
    df_lesion = df_filtered[df_filtered["VoiName(Region) [string]"] == "lesion"]

    # Save the last row’s values
    if not df_lesion.empty:
        last_averaged_suv = df_lesion.iloc[-1]["Averaged [SUV]"]
        last_sd_suv = df_lesion.iloc[-1]["Sd [SUV]"]
    else:
        last_averaged_suv = None
        last_sd_suv = None

    return recon_id, last_averaged_suv, last_sd_suv, df_filtered

def process_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if f.endswith('.voistat')]
    results = []
    for file in files:
        file_path = os.path.join(directory_path, file)
        recon_id, last_averaged_suv, last_sd_suv, _ = process_voistat_file(file_path)
        results.append({
            'recon_id': recon_id,
            'last_averaged_suv': last_averaged_suv,
            'last_sd_suv': last_sd_suv
        })
    
    df = pd.DataFrame(results)
    # Create a new column dividing last_averaged_suv by last_sd_suv
    df['snr'] = df['last_averaged_suv'] / df['last_sd_suv']
    return df

if __name__ == "__main__":
    directory_path = r'C:\Users\DANIE\OneDrive\FAU\Master Thesis\Project\Data\VOI Statistics\PSMA001\VOIstat files and k values'
    result_df = process_directory(directory_path)
    print(result_df)
