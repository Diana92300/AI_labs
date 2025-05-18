import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

file_path = 'augmented_dataset.xlsx'
data = pd.read_excel(file_path)
data = data[['More', 'Race']].dropna()
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_text = vectorizer.fit_transform(data['More']).toarray()
y = data['Race'].factorize()[0]
label_mapping = dict(enumerate(data['Race'].factorize()[1]))

detailed_breed_mapping = {
    "SBI": "Sacred Birman",
    "EUR": "European Shorthair",
    "NR": "Norwegian Forest Cat",
    "MCO": "Maine Coon",
    "BEN": "Bengal",
    "NSP": "Non-Specific",
    "PER": "Persian",
    "ORI": "Oriental Shorthair",
    "BRI": "British Shorthair",
    "Autre": "Other",
    "CHA": "Chartreux",
    "RAG": "Ragdoll",
    "TUV": "Turkish Van",
    "SPH": "Sphynx",
    "SAV": "Savannah"
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_text)

num_classes = len(np.unique(y))
min_class_size = np.min(np.bincount(y))
k_neighbors = min(5, min_class_size - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
y_onehot_resampled = np.zeros((len(y_resampled), num_classes))
y_onehot_resampled[np.arange(len(y_resampled)), y_resampled] = 1
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_onehot_resampled, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
hidden_sizes = [256, 512, 256]
output_size = num_classes
learning_rate = 0.01
epochs = 200

def init_weights(input_size, hidden_sizes, output_size):
    layers = [input_size] + hidden_sizes + [output_size]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1. / layers[i]))
        biases.append(np.zeros(layers[i + 1]))
    return weights, biases

weights, biases = init_weights(input_size, hidden_sizes, output_size)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

def forward_pass(X, weights, biases):
    activations = [X]
    for W, b in zip(weights[:-1], biases[:-1]):
        Z = np.dot(activations[-1], W) + b
        A = relu(Z)
        activations.append(A)

    Z_out = np.dot(activations[-1], weights[-1]) + biases[-1]
    A_out = softmax(Z_out)
    activations.append(A_out)

    return activations

def backward_pass(activations, y_true, weights, biases, learning_rate):
    grads_w = []
    grads_b = []
    m = y_true.shape[0]
    dz = activations[-1] - y_true

    for i in range(len(weights) - 1, -1, -1):
        dw = np.dot(activations[i].T, dz) / m
        db = np.sum(dz, axis=0) / m
        grads_w.insert(0, dw)
        grads_b.insert(0, db)
        if i > 0:
            dz = np.dot(dz, weights[i].T) * relu_derivative(activations[i])

    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_w[i]
        biases[i] -= learning_rate * grads_b[i]
    return weights, biases

for epoch in range(1, epochs + 1):
    activations = forward_pass(X_train, weights, biases)
    loss = cross_entropy_loss(y_train, activations[-1])
    weights, biases = backward_pass(activations, y_train, weights, biases, learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

def predict_description_with_keywords(description):
    description_vector = vectorizer.transform([description]).toarray()
    description_vector_scaled = scaler.transform(description_vector)
    activations = forward_pass(description_vector_scaled, weights, biases)
    predicted_class = np.argmax(activations[-1], axis=1)[0]
    predicted_breed_code = label_mapping[predicted_class]
    detailed_breed = detailed_breed_mapping.get(predicted_breed_code, "Unknown")
    feature_importance = description_vector[0]
    top_indices = np.argsort(feature_importance)[::-1][:5]  # Cele mai relevante 5 cuvinte
    top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    top_scores = feature_importance[top_indices]

    print("Cuvinte cheie care au contribuit la predicție:")
    for word, score in zip(top_keywords, top_scores):
        print(f"- {word}: {score:.4f}")

    return detailed_breed

description="This cat is highly adaptable, enjoying both indoor comforts and occasional outdoor exploration under supervision. It has a playful nature, often engaging with interactive toys or watching birds from the safety of a balcony. While it rarely hunts, it shows keen curiosity about its environment and enjoys bonding moments with its human companions."
predicted_race = predict_description_with_keywords(description)
print(f"\nDescrierea: {description}\nPredicția: {predicted_race}")