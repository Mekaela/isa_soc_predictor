import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
import pickle

# This script creates a trained model file, and should be used on a csv with the following columns:
# Tannual, Pannual, Tillage, CoverCropGroup, GrainCropGroup, OC
# if the dataset is not cleaned, the model will need an imputer added to the pipeline.

def get_data():
    """
    Load the dataset from a CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv('./data/soc_dataset.csv')
    return data

# Identify numeric and categorical columns
numeric_features = ['Tannual', 'Pannual']
categorical_features = ['Tillage', 'CoverCropGroup', 'GrainCropGroup'] # , 'Conservation_Type'

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', dtype='int32'))
])

numeric_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer())
])

preprocessor = ColumnTransformer([
    ('categoricals', categorical_transformer, categorical_features),
    ('numericals', numeric_transformer, numeric_features)
])

pipeline = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('clf', RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)) #RandomForestClassifier(n_estimators=10)) #tree.DecisionTreeClassifier()) #LinearRegression() 
    ]
)

def split_data(data):
    """
    Split the dataset into training and testing sets.
    
    Args:
        data (pd.DataFrame): The dataset to split.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Features and target
    X = data[numeric_features + categorical_features] 
    y = data['OC']
    
    # First split: train+val and test (15% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    
    return X_train, X_test, y_train, y_test

def find_best_model(X_train, y_train):
    """
    Fit the model to the training data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        
    Returns:
        Pipeline: The fitted model pipeline.
    """
    # Define models and parameter grids
    models_params = {
        'RandomForestRegressor': {
            'clf': [RandomForestRegressor(random_state=42)],
            'clf__n_estimators': [10, 50, 100],
            'clf__max_depth': [None, 5, 10]
        },
        'LinearRegression': {
            'clf': [LinearRegression()]
        },
        'Ridge': {
            'clf': [Ridge(alpha=1.0)],
            'clf__alpha': [0.1, 1.0, 10.0]
        }
    }

    best_score = -float('inf')
    best_model = None

    for name, params in models_params.items():
        search = GridSearchCV(pipeline, params, cv=KFold(n_splits=5, shuffle=True, random_state=1), scoring='r2', n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"{name} best CV R^2: {search.best_score_:.3f}")
        if search.best_score_ > best_score:
            best_model = search.best_estimator_
    
    return best_model

def fit_model(X_train, y_train):
    """
    Fit the model to the training data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        
    Returns:
        Pipeline: The fitted model pipeline.
    """
    model= find_best_model(X_train, y_train)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Args:
        model (Pipeline): The trained model.
        filename (str): The filename to save the model.
    """
    # save the soc classification model as a pickle file
    model_pkl_file = filename  

    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)

def main():
    """
    Main function to execute the model training and saving.
    """
    # Load data
    data = get_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Fit the model
    model = fit_model(X_train, y_train)
    
    # Save the model
    save_model(model, './soc_classifier_model.pkl')

if __name__ == "__main__":
    main()
    print("Model training and saving completed successfully.")