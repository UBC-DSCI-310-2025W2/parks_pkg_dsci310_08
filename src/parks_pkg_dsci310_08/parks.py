"""
A module that contains useful functions for parks' ranking data analysis project
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

invalid_characters = r'[<>:"\|*?]'

def create_directory(path: str):
    """
    Creates the directory given by the specified path.
    Purpose: If the directory where the user wants to store their data/file does not exist, this function will create it for them.

    Returns an error if the specified directory name is empty.
    Returns an error if the specified directory name is illegal.

    Parameters:
    ----------
    path: str
        A string representation of the desired path to be created. 
    
    Returns:
    --------
    pathlib.Path
        A path object to the newly created directory.
        
    Examples:
    --------
    >>> from pathlib import Path
    >>> folder_path = "data/processed/predictions"
    >>> created_path = create_directory(folder_path)
    """
    # validating arguments passed in
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if path == "":
        raise ValueError("Path of directory cannot be empty")
    if re.search(invalid_characters, path):
        raise ValueError("Directory name contains at least one illegal character")

    # create directory
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    return path

def find_measurement_last_time(data_frame: pd.DataFrame, time_col: str, group_col: str, measurement_col: str):
    """
    Finds the previous time period's measurement for a given group and adds it as a new column
    called `{measurement_col}_last_time` in the data frame.
    
    This function sorts the data by the group variable and time (i.e., year), 
    then shifts the specified measurement column by one period within each group 
    to find the previous measurement value. It fills any missing values 
    (usually the first period of a group) with the current period's measurement.
    
    We explicitly request that the measurement column does not have any missing data
    as it is the target variable. If there are any missing values in the measurement
    column, the function will raise a ValueError.

    Parameters:
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    time_col : str
        The name of the column representing the time (i.e., 'year') for sorting
    group_col : str
        The name of the group column (i.e., 'city')
    measurement_col : str
        The name of the column containing the measurement value (i.e., `rank`) 
        (should not have NAs).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with one more column that the original DataFrame:
        The returned DataFrame should have this additional column:
        - '{measurement_col}_last_time': 
            the previous time period's measurement for a given group
        
    Examples:
    --------
    >>> import pandas as pd
    >>> data_raw = pd.read_csv('example_data.csv') 
    >>> data_processed = find_measurement_last_time(data_raw, 'year', 'city', 'rank')
    >>> print(data_processed)

    Notes:
    -----
    This function uses the pandas library to perform the task.

    """
    if data_frame[measurement_col].isna().any():
        raise ValueError("The measurement column should not have any missing values.")
    
    # Sort by group and time columns to ensure accurate shifting and ordering
    data_frame_processed = data_frame.sort_values([group_col, time_col])
    
    # Define the new column name 
    new_col_name = f"{measurement_col}_last_time"
    
    # Shift the measurement with respect to time within each group
    data_frame_processed[new_col_name] = data_frame_processed.groupby(group_col)[measurement_col].shift(1)
    
    # Fill NaNs (first instances) with the current time's measurement
    data_frame_processed[new_col_name] = data_frame_processed[new_col_name].fillna(data_frame_processed[measurement_col])
    
    return data_frame_processed

def get_model_coefficients(fitted_pipe: Pipeline) -> pd.DataFrame:
    """
    Extracts feature coefficients from a fitted Ridge regression pipeline and returns
    a DataFrame with clean feature names and coefficients in descending order.
    
    The model pipeline must have two steps – ColumnTransformer and the Ridge model.
    The ColumnTransformer is expected to use OneHotEncoder for categorical features and
    'passthrough' for numerical features, as the function strips the 'onehotencoder__' and
    'passthrough__' prefixes from feature names.
    
    Returns a TypeError if the parameter passed is not a sklearn Pipeline instance.
    Returns a ValueError if the pipeline does not contain 'columntransformer' and/or 'ridge' steps.
    
    Parameters
    ----------
    fitted_pipe : sklearn.pipeline.Pipeline
        A fitted pipeline containing a 'columntransformer' step followed by a 'ridge' step.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'feature' and 'coefficient', sorted by absolute
        coefficient value in descending order, with transformer prefixes removed
        from feature names.
    
    Examples
    --------
    >>> # Given a fitted sklearn Pipeline with a ColumnTransformer and Ridge step
    >>> coef_df = get_model_coefficients(final_pipe)
    >>> coef_df.head()
    
    Notes:
    --------
    This function uses the pandas and sklearn libraries to perform the task
    
    """
    # validating arguments passed in
    if not isinstance(fitted_pipe, Pipeline):
        raise TypeError("fitted_pipe must be a fitted sklearn Pipeline instance.")
    
    if 'columntransformer' not in fitted_pipe.named_steps:
        raise ValueError("Pipeline must contain a 'columntransformer' step.")
    
    if 'ridge' not in fitted_pipe.named_steps:
        raise ValueError("Pipeline must contain a 'ridge' step.")
    
    # extracting feature names and coefficients
    feature_names = fitted_pipe.named_steps['columntransformer'].get_feature_names_out()
    coefficients = fitted_pipe.named_steps['ridge'].coef_
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False)
    
    coef_df['feature'] = (
        coef_df['feature']
        .str.replace('onehotencoder__', '', regex=False)
        .str.replace('passthrough__', '', regex=False)
    )
    # Note: prefixes from other transformers (e.g. 'standardscaler__') will not be removed
    
    return coef_df.reset_index(drop=True)

def split_dataset(data_frame, target_col, test_size=0.2, random_state=None):
    """
    Split a pandas DataFrame into train and test sets by target column.

    Separates the input DataFrame into features (X) and target (y), then splits each 
    into training and test sets using sklearn's train_test_split.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame containing both features and the target column.
    target_col : str
        The name of the column in data_frame to use as the target (y).
    test_size : float, optional
        Proportion of the dataset to include in the test split.
        Must be between 0 and 1 (exclusive). Default is 0.2.
    random_state : int or None, optional
        Controls the shuffling for reproducible output. Default is None.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple (X_train, X_test, y_train, y_test) where:
        - X_train : training features DataFrame
        - X_test  : test features DataFrame
        - y_train : training target Series (as DataFrame with one column)
        - y_test  : test target Series (as DataFrame with one column)

    Raises
    ------
    TypeError
        If data_frame is not a pandas DataFrame.
    ValueError
        If target_col is not a column in data_frame.
    ValueError
        If test_size is not strictly between 0 and 1.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data_processed = pd.read_csv(data_path)
    >>> X_train, X_test, y_train, y_test = split_dataset(
        data_processed, target_col="rank", test_size=0.2, random_state=73
        )
    """
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")

    if target_col not in data_frame.columns:
        raise ValueError(
            f"target_col '{target_col}' is not a column in data_frame. "
            f"Available columns: {list(data_frame.columns)}"
        )

    if not (0 < test_size < 1):
        raise ValueError(
            f"test_size must be between 0 and 1 (exclusive), got {test_size}."
        )

    X = data_frame.drop(columns=[target_col])
    y = data_frame[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
