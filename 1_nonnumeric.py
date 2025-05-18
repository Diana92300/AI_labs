import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

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

def main():
    file_path = r'Data cat personality and predation Cordonnier et al.xlsx'
    sheet_name = 'Data'
    df = load_data(file_path, sheet_name)
    if 'Horodateur' in df.columns:
        df = df.drop(columns=['Horodateur'])
    exclude_columns = ['Plus', 'Race']
    transformed_df, encoders = transform_non_numeric_to_numeric(df, exclude_columns)
    transformed_df = reorder_columns(transformed_df, 'Race')
    output_path = r'Data_non_numeric.xlsx'
    transformed_df.to_excel(output_path, index=False)
    print(f"Transformed dataset saved to {output_path}")

if __name__ == "__main__":
    main()