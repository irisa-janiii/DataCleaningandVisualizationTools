import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data cleaning: Remove duplicates & handle missing values
def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numeric values with mean
    print("Data cleaned successfully.")
    return df

# Visualization: Generate bar and line plots
def visualize_data(df):
    plt.figure(figsize=(10, 5))
    
    # Example: Count plot for a categorical column (modify as per dataset)
    sns.countplot(x=df.columns[0], data=df)
    plt.title("Category Distribution")
    plt.show()

    # Example: Line plot for numerical trends
    if df.select_dtypes(include=['number']).shape[1] > 1:
        df.plot(kind="line", figsize=(10, 5))
        plt.title("Numerical Data Trends")
        plt.show()
    else:
        print("Not enough numeric columns for line plot.")

if __name__ == "__main__":
    file_path = "data/sample_data.csv"  # Make sure to have a sample dataset
    df = load_data(file_path)
    
    if df is not None:
        df = clean_data(df)
        visualize_data(df)
