import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MonthlyCharges'], bins=10, kde=True)
    plt.title('Distribution of Monthly Charges')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['TotalCharges'])
    plt.title('Boxplot of Total Charges')
    plt.xlabel('Total Charges')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MonthlyCharges', y='TotalCharges', data=df)
    plt.title('Scatter Plot of Monthly Charges vs. Total Charges')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.show()