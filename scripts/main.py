from data_loading import load_data
from data_cleaning import clean_data
from data_viz import visualize_data
from feature_engineering import engineer_features
from data_split import split_data
from model_training import train_model
from hyperparameter_tuning import tune_hyperparameters
from evaluate_model import evaluate_model
from feature_importance import save_feature_importance
import joblib

def main():
    # Filepath to the dataset
    filepath = "/workspace/customer-conversion-drivers/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Step 1: Load data
    print("Loading data...")
    df = load_data(filepath)
    
    # Step 2: Clean data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Step 3: Visualize data
    print("Visualizing data...")
    visualize_data(df)
    
    # Step 4: Engineer features
    print("Engineering features...")
    engineer_features(df)
    
    # Step 5: Split data into training and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 6: Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X_train, y_train)
    print("Best parameters found:", best_params)
    
    # Step 7: Train the model
    print("Training model...")
    model = train_model(X_train, y_train, best_params)
    
    # Step 8: Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Step 9: Save feature importance
    feature_names = X_train.columns
    print("Saving feature importance...")
    save_feature_importance(model, feature_names, csv_path="models/feature_importance.csv", img_path="models/feature_importance.png")
    
    # Step 10: Save the trained model
    joblib.dump(model, 'xgboost_churn_model.pkl')
    print("Model saved as xgboost_churn_model.pkl")

if __name__ == "__main__":
    main()