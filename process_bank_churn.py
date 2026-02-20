import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def get_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns, excluding identifiers and target.
    """
    target_col = 'Exited'
    drop_cols = [target_col, 'id', 'CustomerId', 'Surname']
    
    input_cols = [col for col in df.columns if col not in drop_cols]
    numeric_cols = df[input_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df[input_cols].select_dtypes(include='object').columns.tolist()
    
    return numeric_cols, categorical_cols

# --- Imputation ---

def fit_imputers(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[SimpleImputer, SimpleImputer]:
    """
    Fits numeric (median) and categorical (most_frequent) imputers on the dataframe.
    """
    imputer_num = SimpleImputer(strategy='median').fit(df[numeric_cols])
    imputer_cat = SimpleImputer(strategy='most_frequent').fit(df[categorical_cols])
    return imputer_num, imputer_cat

def apply_imputation(df: pd.DataFrame, imputer_num: SimpleImputer, imputer_cat: SimpleImputer, 
                     numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Applies pre-fitted imputers to the dataframe.
    """
    df_copy = df.copy()
    df_copy[numeric_cols] = imputer_num.transform(df_copy[numeric_cols])
    df_copy[categorical_cols] = imputer_cat.transform(df_copy[categorical_cols])
    return df_copy

# --- Encoding ---

def fit_encoder(df: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fits a OneHotEncoder on the categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[categorical_cols])
    return encoder

def apply_encoding(df: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Applies OneHotEncoding and adds new encoded columns to the dataframe.
    """
    df_copy = df.copy()
    encoded_data = encoder.transform(df_copy[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    # Create a DataFrame with encoded columns and merge
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_copy.index)
    return pd.concat([df_copy, encoded_df], axis=1)

# --- Scaling ---

def fit_scaler(df: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fits a MinMaxScaler on the numeric columns.
    """
    return MinMaxScaler().fit(df[numeric_cols])

def apply_scaling(df: pd.DataFrame, scaler: MinMaxScaler, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Applies pre-fitted MinMaxScaler to the numeric columns.
    """
    df_copy = df.copy()
    df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])
    return df_copy

# --- Main Pipelines ---

def preprocess_data(df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Full training pipeline: splits data, fits all processors on train, and transforms both train and val.
    """
    target_col = 'Exited'
    numeric_cols, categorical_cols = get_feature_cols(df)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[target_col])
    
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    # 1. Impute
    imputer_num, imputer_cat = fit_imputers(train_df, numeric_cols, categorical_cols)
    train_df = apply_imputation(train_df, imputer_num, imputer_cat, numeric_cols, categorical_cols)
    val_df = apply_imputation(val_df, imputer_num, imputer_cat, numeric_cols, categorical_cols)

    # 2. Encode
    encoder = fit_encoder(train_df, categorical_cols)
    train_df = apply_encoding(train_df, encoder, categorical_cols)
    val_df = apply_encoding(val_df, encoder, categorical_cols)
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    # 3. Scale
    scaler = None
    if scaler_numeric:
        scaler = fit_scaler(train_df, numeric_cols)
        train_df = apply_scaling(train_df, scaler, numeric_cols)
        val_df = apply_scaling(val_df, scaler, numeric_cols)

    # Select final features
    final_cols = numeric_cols + encoded_cols
    
    return {
        'train_X': train_df[final_cols],
        'train_y': train_targets,
        'val_X': val_df[final_cols],
        'val_y': val_targets,
        'imputer_num': imputer_num,
        'imputer_cat': imputer_cat,
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols
    }

def preprocess_new_data(
    df: pd.DataFrame,
    imputer_num: SimpleImputer,
    imputer_cat: SimpleImputer,
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler],
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Inference pipeline: applies pre-fitted processors to new data (e.g., test.csv).
    """
    # 1. Impute
    df = apply_imputation(df, imputer_num, imputer_cat, numeric_cols, categorical_cols)
    
    # 2. Encode
    df = apply_encoding(df, encoder, categorical_cols)
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    # 3. Scale
    if scaler is not None:
        df = apply_scaling(df, scaler, numeric_cols)
        
    return df[numeric_cols + encoded_cols]