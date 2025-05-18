import pandas as pd
import random
import google.generativeai as genai

def load_dataset(file_path):
    return pd.read_excel(file_path, sheet_name='Sheet1')

def generate_prompt(df, race, attribute_descriptions, num_attributes=10):
    race_data = df[df['Race'] == race]
    if race_data.empty:
        return f"No data available for the race: {race}."

    numeric_columns = [col for col in race_data.columns if pd.api.types.is_numeric_dtype(race_data[col]) and col in attribute_descriptions]
    attributes = race_data[numeric_columns].mean().nlargest(num_attributes).index.tolist()

    prompt_content = f"The following is a detailed description of the {race} breed based on several characteristics:\n\n"
    for attr in attributes:
        description = attribute_descriptions[attr]
        if attr in race_data.columns and pd.api.types.is_numeric_dtype(race_data[attr]):
            value = race_data[attr].mean()
        else:
            value = "N/A"
        prompt_content += f"- {attr}: {description} (Value: {value}).\n"

    prompt_content += "\nUsing this information, describe the breed in detail without mentioning the value of the description. I want a few propozition and the description should only touch on the main points. Don't go astray."

    print("Selected Attributes and Scores:")
    for attr in attributes:
        if attr in race_data.columns and pd.api.types.is_numeric_dtype(race_data[attr]):
            score = race_data[attr].mean()
            print(f"{attr}: {score}")
    return prompt_content

api_key = "..."
genai.configure(api_key=api_key)

def call_gemini_api(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    if response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    else:
        return "No description generated."

def generate_comparison_prompt(description1, description2, race1, race2):
    comparison_prompt = (
        f"Here are two detailed descriptions of the cat breeds {race1} and {race2} respectively:\n\n"
        f"Description of {race1}:\n{description1}\n\n"
        f"Description of {race2}:\n{description2}\n\n"
        "Based on these descriptions, identify the main similarities and differences between the two breeds. "
        "Provide a concise comparison that highlights their unique traits as well as common characteristics."
    )
    return comparison_prompt

attribute_descriptions = {
    "Sexe": "Gender of the cat, can be male or female.",
    "Age": "Age of the cat categorized into different ranges: less than 1 year, 1-2 years, 2-10 years, and more than 10 years.",
    "Race": "The breed of the cat, including options like Bengal, Birman, European, Maine Coon, etc.",
    "Nombre": "Number of cats in the household.",
    "Logement": "Type of housing, such as apartment without balcony, apartment with balcony, house, or individual house.",
    "Zone": "Type of area where the cat lives: urban, peri-urban, or rural.",
    "Ext": "How much time the cat spends outdoors daily, ranging from none to all day.",
    "Obs": "How much time the owner spends daily with the cat on activities like observation, petting, and games.",
    "Abondance": "Abundance of natural areas (trees, bushes, grass) around the residence.",
    "PredOiseau": "Frequency of the cat capturing birds, ranging from never to very often.",
    "PredMamm": "Frequency of the cat capturing small mammals, ranging from never to very often."
}

file_path = "augmented_dataset.xlsx"
df = load_dataset(file_path)

selected_race1 = "SBI"
selected_race2 = "EUR"
prompt1 = generate_prompt(df, selected_race1, attribute_descriptions)
prompt2 = generate_prompt(df, selected_race2, attribute_descriptions)

description1 = call_gemini_api(prompt1)
description2 = call_gemini_api(prompt2)

comparison_prompt = generate_comparison_prompt(description1, description2, selected_race1, selected_race2)

comparison_response = call_gemini_api(comparison_prompt)

print("Description of Race 1:")
print(description1)
print("\nDescription of Race 2:")
print(description2)
print("\nComparison Prompt:")
print(comparison_prompt)
print("\nGemini API Comparison Response:")
print(comparison_response)
