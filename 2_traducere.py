import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
def load_translation_model():
    try:
        model_name = "Helsinki-NLP/opus-mt-fr-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Eroare la încărcarea modelului de traducere: {e}")

def translate_text(texts, tokenizer, model):
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    except Exception as e:
        print(f"Eroare la traducerea textului: {texts}. Eroare: {e}")
        return texts

def translate_column_names(columns, tokenizer, model):
    return [translate_text([col], tokenizer, model)[0] for col in columns]

def translate_specific_column(df, column_name, tokenizer, model):
    if column_name in df.columns:
        print(f"Traducerea coloanei '{column_name}'...")
        df[column_name] = df[column_name].apply(
            lambda x: translate_text([x], tokenizer, model)[0] if isinstance(x, str) else x
        )
    else:
        print(f"Coloana '{column_name}' nu a fost găsită în DataFrame.")
    return df

def save_data(df, output_path):
    try:
        df.to_excel(output_path, index=False)
        print(f"Fișierul tradus a fost salvat în: {output_path}")
    except PermissionError:
        raise PermissionError(
            f"Nu s-a putut salva fișierul la {output_path}. Închideți alte aplicații care folosesc acest fișier și încercați din nou.")
    except Exception as e:
        raise RuntimeError(f"Eroare la salvarea fișierului: {e}")


def main():
    input_file_path = r'Data_non_numeric.xlsx'
    sheet_name = 'Sheet1'
    output_file_path = r'Data_translate.xlsx'
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Fișierul de intrare nu există: {input_file_path}")
    try:
        df = pd.read_excel(input_file_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"Eroare la încărcarea fișierului Excel: {e}")
    tokenizer, model = load_translation_model()
    print("Traducerea numelor coloanelor...")
    original_columns = list(df.columns)
    translated_columns = translate_column_names(original_columns, tokenizer, model)
    df.columns = translated_columns
    df = translate_specific_column(df, "More", tokenizer, model)
    save_data(df, output_file_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Eroare în timpul rulării scriptului: {e}")