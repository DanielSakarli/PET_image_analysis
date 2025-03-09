import os
import pandas as pd


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
    directory_path = r'C:\Users\DANIE\OneDrive\FAU\Master Thesis\Project\Data\VOI Statistics\PSMA010\VOIstat files and k values'
    result_df = process_directory(directory_path)
    print(result_df)