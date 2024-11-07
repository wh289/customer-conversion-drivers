from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBClassifier

from data_loading import load_data
from data_cleaning import clean_data
from feature_engineering import engineer_features
from model_training import train_model
from evaluate_model import evaluate_model
from feature_importance import save_feature_importance
from hyperparameter_tuning import tune_hyperparameters
from data_split import split_data


class ChurnFlow(FlowSpec):

    # Define parameters
    data_path = Parameter('data_path', default="/workspace/customer-conversion-drivers/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_path = Parameter('model_path', default="models/xgboost_churn_model.pkl")
    importance_path = Parameter('importance_path', default="models/feature_importance.csv")

    @step
    def start(self):
        """
        Step 1: Load the data.
        """
        print("Loading data...")
        self.df = load_data(self.data_path)
        self.next(self.clean_data)
    
    @step
    def clean_data(self):
        """
        Step 2: Clean the data.
        """
        print("Cleaning data...")
        self.df = clean_data(self.df)
        self.next(self.engineer_features)
    
    @step
    def engineer_features(self):
        """
        Step 3: Engineer features.
        """
        print("Engineering features...")
        self.df = engineer_features(self.df)
        self.next(self.split_data)
    
    @step
    def split_data(self):
        """
        Step 4: Split data into training and test sets.
        """
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)
        self.next(self.tune_hyperparameters)
    
    @step
    def tune_hyperparameters(self):
        """
        Step 5: Tune hyperparameters.
        """
        print("Tuning hyperparameters...")
        self.best_params = tune_hyperparameters(self.X_train, self.y_train)
        print("Best parameters found:", self.best_params)
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Step 6: Train the model.
        """
        print("Training model...")
        self.model = train_model(self.X_train, self.y_train, self.best_params)
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        """
        Step 7: Evaluate the model.
        """
        print("Evaluating model...")
        evaluate_model(self.model, self.X_test, self.y_test)
        self.next(self.save_feature_importance)
    
    @step
    def save_feature_importance(self):
        """
        Step 8: Save feature importance as a table.
        """
        feature_names = self.X_train.columns
        print("Saving feature importance...")
        save_feature_importance(self.model, feature_names, self.importance_path)
        self.next(self.save_model)
    
    @step
    def save_model(self):
        """
        Step 9: Save the trained model.
        """
        print(f"Saving model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)
        print("Model saved.")
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("Churn prediction pipeline completed!")


if __name__ == "__main__":
    ChurnFlow()