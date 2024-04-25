import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time

# Define model parameters
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'decisiontreeclassifier__max_depth': [None, 5, 10, 20],
            'decisiontreeclassifier__min_samples_split': [2, 5, 10],
            'decisiontreeclassifier__min_samples_leaf': [1, 2, 4]
        }
    }
}

# Function to compare models and return the best one
def model_compare(X_train, y_train):
    scores = []
    best_estimators = {}
    for algo, mp in model_params.items():
        pipe = make_pipeline(StandardScaler(), mp['model'])
        clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
        st.write(f"Training {algo} model...")
        progress_bar = st.progress(0)
        for i in range(101):
            time.sleep(0.05)  # Simulate training time
            progress_bar.progress(i)
        clf.fit(X_train, y_train)
        scores.append({
            'model': algo,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'best_estimator': clf.best_estimator_
        })
        best_estimators[algo] = clf.best_estimator_
    models = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params', 'best_estimator'])
    return best_estimators, models

# Function to preprocess data and make predictions
def makePrediction(df, userInputs, best_model):
    x_cols = df.iloc[:, :-1]

    # Handling missing data for numerical columns
    numerical_columns = x_cols.select_dtypes(include=['int64', 'float64'])
    for column in numerical_columns.columns:
        x_cols.loc[:, column].fillna(x_cols[column].mean(), inplace=True)

    # Handling missing data and encoding categorical columns
    categorical_columns = x_cols.select_dtypes(include=['object'])
    for column in categorical_columns.columns:
        x_cols.loc[:, column].fillna(x_cols[column].mode().iloc[0], inplace=True)
        x_cols = pd.get_dummies(x_cols, columns=[column])

    # Encode boolean columns
    x_cols = x_cols.replace({True: 1, False: 0})

    # Feature Scaling
    sc = StandardScaler()
    x_cols_scaled = sc.fit_transform(x_cols)

    # Model Training
    best_model.fit(x_cols_scaled, df.iloc[:, -1])

    # Preprocessing the user input
    user_inputs_dict = {}
    for i, col in enumerate(df.columns[:-1]):
        user_inputs_dict[col] = float(userInputs[i]) if i < len(userInputs) else 0

    user_inputs_df = pd.DataFrame([user_inputs_dict])
    user_inputs_scaled = sc.transform(user_inputs_df)

    # Predicting the result
    prediction = best_model.predict(user_inputs_scaled)
    return prediction[0]

# Load CSV file and prepare data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=[df.columns[-1]])
    return df

# Main Streamlit app
def main():
    st.title('Streamlined Model Selection & Deployment: Automated Comparison for Optimal Predictive Performance')

    # Sidebar: Upload CSV file and get user inputs
    st.sidebar.title('Upload CSV and Input Values')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("CSV Data:")
        st.write(df.head())

        # Get user inputs
        user_inputs = []
        for col in df.columns[:-1]:
            user_input = st.sidebar.text_input(f"Enter {col} value:")
            user_inputs.append(user_input)

        # Train and Predict
        if st.sidebar.button("Train and Predict"):
            st.sidebar.write("Training in progress...")
            best_estimators, models = model_compare(df.iloc[:, :-1], df.iloc[:, -1])
            best_model_name = models.loc[models['best_score'].idxmax()]['model']
            best_model = best_estimators[best_model_name]
            prediction = makePrediction(df, user_inputs, best_model)

            

            # Display model scores in tabular format
            st.write("Model Comparison:")
            st.write(models)
            
            st.write(f"Best Model: {best_model_name}")

            # Plotting accuracy graph
            plt.figure(figsize=(10, 6))
            plt.bar(models['model'], models['best_score'], color='blue')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy Comparison')
            plt.ylim(0, 1)
            st.pyplot(plt)
            
            # Display best model and prediction
            
            st.markdown(f'<h1 style="color:green;">Used Model: {best_model_name}</h1>', unsafe_allow_html=True)

            st.markdown(f'<h1 style="color:blue;">Prediction: {prediction}</h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
