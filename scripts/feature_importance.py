# feature_importance.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_feature_importance(model, feature_names, csv_path="models/feature_importance.csv", img_path="models/feature_importance.png"):
    """
    Saves the feature importance data as a CSV and generates a bar plot saved as an image.

    Parameters:
    - model: The trained XGBoost model.
    - feature_names: List of feature names used in training the model.
    - csv_path: Path where the feature importance CSV file will be saved.
    - img_path: Path where the feature importance plot image will be saved.
    """
    # Ensure the directory for csv_path and img_path exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # Extract feature importance using get_booster().get_score()
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    # Convert the importance dictionary to a DataFrame
    importance_df = pd.DataFrame(
        {'Feature': list(importance_dict.keys()), 'Importance': list(importance_dict.values())}
    )
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Save the DataFrame as a CSV file
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance data saved to {csv_path}")
    
    # Plot and save the feature importance as an image
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance from XGBoost Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(img_path)
    print(f"Feature importance plot saved to {img_path}")