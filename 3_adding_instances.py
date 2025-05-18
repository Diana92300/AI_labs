import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

def load_data(file_path):
    return pd.read_excel(file_path, engine='openpyxl')
def transform_non_numeric_to_numeric(df, exclude_columns):
    label_encoders = {}
    non_numeric_columns = df.select_dtypes(include=['object']).columns.difference(exclude_columns)

    for col in non_numeric_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

def reorder_columns(df, target_column):
    columns = list(df.columns)
    columns.remove(target_column)
    columns.insert(1, target_column)
    return df[columns]

def augment_data(df, target_column):
    X = df.drop(columns=[target_column, 'More'])
    y = df[target_column]
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    repetitions = len(df_resampled) // len(df) + 1
    df_resampled['More'] = np.tile(df['More'].values, repetitions)[:len(df_resampled)]
    df_resampled = reorder_columns(df_resampled, target_column)

    return df_resampled

def main():
    file_path = 'Data_translate.xlsx'
    df = load_data(file_path)
    exclude_columns = ['Race', 'More']
    transformed_df, encoders = transform_non_numeric_to_numeric(df, exclude_columns)
    transformed_df = reorder_columns(transformed_df, 'Race')
    augmented_df = augment_data(transformed_df, 'Race')
    output_path = 'augmented_dataset.xlsx'
    augmented_df.to_excel(output_path, index=False)
    print(f"Augmented dataset saved to {output_path}")

if __name__ == "__main__":
    main()
