import os
import numpy as np
import librosa
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# feature extraction

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# loads dataset (all folders but test)

def load_dataset(base_dir="."):
    X = []
    y = []

    for folder in os.listdir(base_dir):
        if folder in ["Test", "__pycache__"]:
            continue

        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    path = os.path.join(folder_path, file)
                    features = extract_mfcc(path)
                    X.append(features)
                    y.append(folder)

    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset(".")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# building neural network

model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
overall_accuracy = test_accuracy * 100

# prediction function

def predict_audio(audio_path):
    features = extract_mfcc(audio_path)
    features = features.reshape(1, -1)

    probs = model.predict(features)[0]
    predicted_index = np.argmax(probs)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(probs) * 100

    return predicted_label, confidence

# tkinter interface

root = tk.Tk()
root.title("Accent Classification Neural Network")
root.geometry("900x600")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# tkinter accuracy graph

accuracy_tab = ttk.Frame(notebook)
notebook.add(accuracy_tab, text="Accuracy")

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(history.history["accuracy"])
ax1.plot(history.history["val_accuracy"])
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend(["Train", "Validation"])

canvas1 = FigureCanvasTkAgg(fig1, master=accuracy_tab)
canvas1.draw()
canvas1.get_tk_widget().pack(fill="both", expand=True)

# tkinter loss graph

loss_tab = ttk.Frame(notebook)
notebook.add(loss_tab, text="Loss")

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(history.history["loss"])
ax2.plot(history.history["val_loss"])
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(["Train", "Validation"])

canvas2 = FigureCanvasTkAgg(fig2, master=loss_tab)
canvas2.draw()
canvas2.get_tk_widget().pack(fill="both", expand=True)

# tkinter audio prediction

prediction_tab = ttk.Frame(notebook)
notebook.add(prediction_tab, text="Audio Prediction")

title_label = tk.Label(
    prediction_tab,
    text="Accent Classification System",
    font=("Arial", 16)
)
title_label.pack(pady=10)

accuracy_label = tk.Label(
    prediction_tab,
    text=f"Overall Model Accuracy: {overall_accuracy:.2f}%",
    font=("Arial", 12),
    fg="blue"
)
accuracy_label.pack(pady=5)

result_text = tk.StringVar()

def choose_file():
    file_path = filedialog.askopenfilename(
        initialdir="./Test",
        title="Select Test Audio",
        filetypes=(("WAV files", "*.wav"),)
    )

    if file_path:
        label, confidence = predict_audio(file_path)

        result_text.set(
            f"Prediction: {label}\n"
            f"Confidence: {confidence:.2f}%"
        )

btn = tk.Button(
    prediction_tab,
    text="Choose Test Audio",
    command=choose_file,
    width=25
)
btn.pack(pady=20)

result_label = tk.Label(
    prediction_tab,
    textvariable=result_text,
    font=("Arial", 14)
)
result_label.pack(pady=20)

root.mainloop()


