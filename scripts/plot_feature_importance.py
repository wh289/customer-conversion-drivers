import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for an XGBoost model.

    Parameters:
    - model: The trained XGBoost model
    - feature_names: List of feature names used in training the model
    """
    # Extract feature importance using get_booster().get_score()
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    # Convert the importance dictionary to a DataFrame for easier plotting
    importance_df = pd.DataFrame(
        {'Feature': list(importance_dict.keys()), 'Importance': list(importance_dict.values())}
    )
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', hue=None, legend=False)
    plt.title('Feature Importance from XGBoost Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

     # Save the plot to a file instead of displaying it
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved as feature_importance.png")