# Catology: AI-Powered Cat Breed Identification

Catology is an AI-driven system designed to identify a cat’s breed based on textual and numerical attributes. By combining data preprocessing, natural language processing, machine learning, and data visualization, Catology automates the pipeline from raw survey data to human-readable breed summaries and comparisons.

## Features

* **Dataset Augmentation & Translation**

  * Translates  French entities into English.
  * Expands the dataset using SMOTE to balance breed classes.

* **Natural Language Processing (NLP)**

  * Parses survey text to extract relevant cat attributes.
  * Generates concise, human-readable breed summaries.
  * Enables natural-language comparisons between breeds.

* **Data Preprocessing & Validation**

  * Handles missing values, duplicates, and inconsistent entries.
  * Converts categorical fields into numeric codes.
  * Reports instance counts per breed and unique attribute values.

* **Statistical Analysis & Visualization**

  * Creates histograms, box plots, and distribution graphs for each attribute.
  * Identifies correlations between attributes and breeds.

* **Machine Learning Models**

  * Transforms categorical and textual data into numerical features.
  * Trains and evaluates classifiers (neural networks, tree-based models) to predict breeds.
  * Reports metrics: precision, recall,  and confusion matrices.

* **AI-Driven Description & Comparison**

  * Uses Hugging Face / Google Gemini APIs to generate detailed breed descriptions.
  * Creates side-by-side breed comparisons in natural language.

## Technologies & Tools

* **Language:** Python 3.8+
* **Data Processing:** pandas, numpy, openpyxl
* **NLP & Translation:** NLTK, spaCy, Hugging Face Transformers, MarianMT
* **Machine Learning:** scikit-learn, imbalanced-learn, PyTorch
* **Visualization:** matplotlib, seaborn


## Results & Interpretation

* **Data Insights:** Summaries of key breed traits and attribute distributions.
* **Model Performance:** Classification reports and confusion matrices for each model.
* **Visualization:** Automatically generated plots showcasing data patterns.
* **AI Summaries:** Sample breed descriptions and comparisons in plain English.


