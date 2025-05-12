import pyreadr
import pandas as pd
import numpy as np
import hashlib

def anonymous_head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the first n rows of the DataFrame with patient_id anonymized using MD5 hash.
    Since the patient_id has to be private and handled carefully, I use this function in this project.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'patient_id' column.
        n (int): Number of rows to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with anonymized 'patient_id' values.
    """
    if 'patient_id' not in df.columns:
        raise ValueError("The DataFrame must contain a 'patient_id' column.")
    
    head_df = df.head(n).copy()
    
    # Convert patient_id to anonymous values
    def anonymize_id(pid):
        return hashlib.md5(str(pid).encode()).hexdigest()[:8]
    
    head_df['patient_id'] = head_df['patient_id'].apply(anonymize_id)
    return head_df

def data_load(file_path:str) -> pd.DataFrame:
    """
    Load data from an R file using pyreadr and return it as a pandas DataFrame.
    """
    r_data = pyreadr.read_r(file_path)
    data_list = list(r_data.values())[0]
    return data_list


def get_info_matrix(df:pd.DataFrame, medication_oh:dict, start_year:int=2019, end_year:int=2024, bins_method:str="half_year") -> pd.DataFrame:
    """
    Generate an information matrix indicating medication patterns across specified time bins.
    """
    # Extract the necessary informations 
    matrix_df = df[['patient_id', 'start_date', 'cls.short']]
    matrix_df = matrix_df.rename(columns={'cls.short': 'medications'})

    # Convert to 'datetime' type
    matrix_df['start_date'] = pd.to_datetime(matrix_df['start_date'], format='%Y%m%d')

    # Sort the date by patient
    mask = matrix_df['start_date'].dt.year.between(start_year, end_year)
    matrix_df = matrix_df.loc[mask].sort_values(['patient_id', 'start_date'])

    # Make the list of medications from one patient on start_date
    # ex. If the patient got two medications on one day, it makes the medications as a list.
    matrix_df = matrix_df.groupby(['patient_id', 'start_date'], as_index=False).agg({'medications': list})


    year_diff = matrix_df['start_date'].dt.year - start_year
    num_bins = 0
    if bins_method == "year":
        num_bins = end_year - start_year + 1
        matrix_df["bin"] = year_diff
    elif bins_method == "month":
        num_bins = (end_year - start_year + 1) * 12
        matrix_df["bin"] = year_diff * 12 + matrix_df["start_date"].dt.month - 1
    elif bins_method == "half_year":
        num_bins = (end_year - start_year + 1) * 2
        matrix_df["bin"] = year_diff * 2 + (matrix_df["start_date"].dt.month > 6).astype(int)
    elif bins_method == "quarter":
        num_bins = (end_year - start_year + 1) * 4
        matrix_df["bin"] = year_diff * 4 + (matrix_df["start_date"].dt.month - 1) // 3
    else:
        raise ValueError("Invalid bins_method. Choose from 'year', 'month', 'half year', or 'quarter'.")

    eye = np.eye(num_bins, dtype=int)

    # OneHotEncoding
    matrix_df["bin_mat"] = matrix_df["bin"].apply(
            lambda x: eye[x]
        )
    matrix_df["med_mat"] = matrix_df["medications"].apply(
            lambda meds: np.sum([np.array(medication_oh[med]) for med in meds], axis=0)
        )

    # Create the OneHotEncoded matrix
    matrix_df["info_mat"] = matrix_df.apply(
            lambda row: row["bin_mat"][:, None]*row["med_mat"], axis=1
        )
    
    # Drop unnecessary columns, after I created final matrix
    matrix_df = matrix_df.drop(["start_date", "medications", "bin", "bin_mat", "med_mat"], axis=1)

    # Combine the separated patient informations
    matrix_df = matrix_df.groupby('patient_id').agg({'info_mat': lambda x: np.sum(np.stack(x), axis=0)}).reset_index()


    cls = df[["patient_id", "cls.short"]].groupby("patient_id", as_index=False).agg(set).rename(columns={"cls.short": "all_medications"})

    matrix_df = pd.merge(matrix_df, cls, on="patient_id")

    matrix_df["info_mat"] = matrix_df['info_mat'].apply(lambda x: x.astype(np.int16).tolist())

    return matrix_df


def convert_medications(df:pd.DataFrame, column:str, old:str, new:str) -> pd.DataFrame:
    """
    Replace old medication names with new names in a specified DataFrame column.
    """
    df[column] = df[column].replace(old, new)
    return df


def medication_one_hot_encoding(df:pd.DataFrame) -> dict:
    """
    Create a dictionary of one-hot encodings for each unique medication in the 'cls.short' column.
    """
    if "cls.short" not in df.columns:
        raise KeyError("The data must contains the \"cls.short\" columns")
    med_oh = dict()
    unique_vals = df["cls.short"].unique()
    for i, vals in enumerate(unique_vals):
        oh = [0]*len(unique_vals)
        oh[i] = 1
        med_oh[vals] = oh

    return med_oh


def save_columns_as_csv(df:pd.DataFrame, columns:list, file_name:str):
    """
    Save specified columns of a DataFrame as a CSV file.
    """
    df = df[columns]
    df.to_csv(f"data/extracted/{file_name}.csv", index=False)


def save_one_hot_encoded_medications(med_oh:dict, file_name:str):
    """
    Save the one-hot encoded medication dictionary as a CSV file.
    """
    df = pd.DataFrame.from_dict(med_oh, orient='index')
    df.to_csv(f"data/extracted/{file_name}.csv")


def get_random_patients(df:pd.DataFrame, num_patients:int=10, random_state:int=42):
    """
    Randomly select a specified number of patients from the DataFrame.
    """
    if df.shape[0] == num_patients:
        return df
    return df.sample(n=min(num_patients, len(df)), random_state=random_state).reset_index(drop=True)

