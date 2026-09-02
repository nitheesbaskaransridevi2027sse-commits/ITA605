import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = None
model = None
X_test = None
y_test = None
label_encoders = {}

def load_dataset():
    global df
    try:
        df = pd.read_csv("agricultural_data.csv")
        print("\nDataset loaded successfully.")
        print("Number of records:", len(df))
        print("Number of attributes:", len(df.columns))
        print("\nColumns:")
        print(df.columns.tolist())
    except FileNotFoundError:
        print("\nError: agricultural_data.csv was not found.")
        print("Place the CSV file in the same folder as this Python program.")

def display_dataset():
    if df is None:
        print("\nPlease load the dataset first.")
        return
    print("\nFirst 10 records:")
    print(df.head(10))
    print("\nDataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

def preprocess_data():
    global df
    if df is None:
        print("\nPlease load the dataset first.")
        return
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())
    df = df.drop_duplicates()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    print("\nPreprocessing completed.")
    print("Duplicate records removed.")
    print("Missing values handled.")
    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())

def statistical_analysis():
    if df is None:
        print("\nPlease load the dataset first.")
        return
    print("\nStatistical Analysis:")
    print(df.describe(include="all"))

def pandas_analysis():
    if df is None:
        print("\nPlease load the dataset first.")
        return
    print("\nPandas Data Analysis")
    print("\nDataset information:")
    print(df.info())
    print("\nSorted data:")
    print(df.head().sort_index())
    print("\nNumber of unique values:")
    print(df.nunique())
    print("\nGrouping example:")
    object_columns = df.select_dtypes(include="object").columns
    if len(object_columns) > 0:
        column = object_columns[0]
        print(df.groupby(column).size())

def prepare_features():
    global df
    if df is None:
        print("\nPlease load the dataset first.")
        return None, None

    print("\nAvailable columns:")
    for i, column in enumerate(df.columns):
        print(i + 1, ".", column)

    target = input("\nEnter the TARGET column name: ").strip()

    if target not in df.columns:
        print("Invalid target column.")
        return None, None

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    for column in X.columns:
        if X[column].dtype == "object":
            if column not in label_encoders:
                label_encoders[column] = LabelEncoder()
                X[column] = label_encoders[column].fit_transform(X[column].astype(str))
            else:
                X[column] = label_encoders[column].transform(X[column].astype(str))

    if y.dtype == "object":
        if "target" not in label_encoders:
            label_encoders["target"] = LabelEncoder()
            y = label_encoders["target"].fit_transform(y.astype(str))
        else:
            y = label_encoders["target"].transform(y.astype(str))

    return X, y

def train_decision_tree():
    global model, X_test, y_test

    X, y = prepare_features()

    if X is None:
        return

    if len(y.unique()) < 2:
        print("\nError: Target variable must contain at least two classes.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\nDecision Tree trained successfully.")
    print("\nModel Evaluation:")
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1-Score :", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

def predict_risk():
    if model is None:
        print("\nPlease train the Decision Tree first.")
        return

    print("\nEnter values for the required features.")

    feature_names = model.feature_names_in_
    input_data = []

    for feature in feature_names:
        while True:
            try:
                value = float(input("Enter " + feature + ": "))
                input_data.append(value)
                break
            except ValueError:
                print("Please enter a numerical value.")

    prediction = model.predict([input_data])[0]

    if "target" in label_encoders:
        try:
            result = label_encoders["target"].inverse_transform([prediction])[0]
        except:
            result = prediction
    else:
        result = prediction

    print("\nPredicted Agricultural Risk / Yield Category:", result)

def show_confusion_matrix():
    if model is None or X_test is None:
        print("\nPlease train the Decision Tree first.")
        return

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Agricultural Risk Classification - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()

def show_decision_tree():
    if model is None:
        print("\nPlease train the Decision Tree first.")
        return

    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=model.feature_names_in_,
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree for Agricultural Risk Classification")
    plt.show()

def show_visualization():
    if df is None:
        print("\nPlease load the dataset first.")
        return

    numerical_columns = df.select_dtypes(include=np.number).columns

    if len(numerical_columns) < 2:
        print("\nNot enough numerical columns for visualization.")
        return

    x_column = numerical_columns[0]
    y_column = numerical_columns[1]

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(x_column + " vs " + y_column)
    plt.show()

def feature_importance():
    if model is None:
        print("\nPlease train the Decision Tree first.")
        return

    importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance)

    plt.figure(figsize=(10, 5))
    importance.plot(kind="bar")
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n==============================================")
        print(" INTELLIGENT AGRICULTURAL RISK SYSTEM")
        print("==============================================")
        print("1. Load Agricultural Dataset")
        print("2. Display Dataset")
        print("3. Perform Data Preprocessing")
        print("4. Perform Statistical Analysis")
        print("5. Perform Pandas Data Analysis")
        print("6. Train Decision Tree")
        print("7. Predict Agricultural Risk")
        print("8. Evaluate Model")
        print("9. Display Confusion Matrix")
        print("10. Display Decision Tree")
        print("11. Display Feature Importance")
        print("12. Display Visualization")
        print("13. Exit")
        print("==============================================")

        choice = input("Enter your choice: ")

        if choice == "1":
            load_dataset()
        elif choice == "2":
            display_dataset()
        elif choice == "3":
            preprocess_data()
        elif choice == "4":
            statistical_analysis()
        elif choice == "5":
            pandas_analysis()
        elif choice == "6":
            train_decision_tree()
        elif choice == "7":
            predict_risk()
        elif choice == "8":
            if model is None:
                print("\nPlease train the Decision Tree first.")
            else:
                predictions = model.predict(X_test)
                print("\nAccuracy :", round(accuracy_score(y_test, predictions) * 100, 2), "%")
                print("Precision:", round(precision_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2), "%")
                print("Recall   :", round(recall_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2), "%")
                print("F1-Score :", round(f1_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2), "%")
        elif choice == "9":
            show_confusion_matrix()
        elif choice == "10":
            show_decision_tree()
        elif choice == "11":
            feature_importance()
        elif choice == "12":
            show_visualization()
        elif choice == "13":
            print("\nProgram terminated.")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()