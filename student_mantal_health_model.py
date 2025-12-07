


!pip install kagglehub

import kagglehub
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam




path = kagglehub.dataset_download("zubairdhuddi/student-dataset")
print("Dataset downloaded at:", path)

df = pd.read_csv(path + "/student_depression_dataset.csv")
print(" Sample Data:")
print(df.head())



df = df.applymap(lambda x: str(x).strip().replace("'", "").replace('"', '') if isinstance(x, str) else x)


df = df.drop(columns=["id"], errors="ignore")

df = df.dropna()

TARGET = "Depression"

X = df.drop(columns=[TARGET])
y = df[TARGET]


if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "label_encoder.pkl")
    print(" Label encoder saved.")
else:
    label_encoder = None



categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)



preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

joblib.dump(preprocessor, "preprocessor.pkl")
print("\n✔ Preprocessor saved.")


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()



callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)



loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")




model.save("mental_health_model.keras")
print("Model saved as mental_health_model.keras")


def predict_student(data_dict):
    """Predict depression risk from raw input."""
    df_input = pd.DataFrame([data_dict])
    pre = joblib.load("preprocessor.pkl")
    loaded_model = model  # already loaded in memory

    processed = pre.transform(df_input)

    prob = loaded_model.predict(processed)[0][0]
    prediction = int(prob > 0.5)

    return {"probability": float(prob), "prediction (0=No, 1=Yes)": prediction}



example = {
    'Gender': 'Male',
    'Age': 22,
    'City': 'Delhi',
    'Profession': 'Student',
    'Academic Pressure': 3,
    'Work Pressure': 0,
    'CGPA': 7.5,
    'Study Satisfaction': 4,
    'Job Satisfaction': 2,
    'Sleep Duration': '5-6 hours',
    'Dietary Habits': "Healthy",
    'Degree': "B.Tech",
    'Have you ever had suicidal thoughts ?': "No",
    'Work/Study Hours': 4,
    'Financial Stress': 2,
    'Family History of Mental Illness': "No"
}

print(" Example Prediction:", predict_student(example))
