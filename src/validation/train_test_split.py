import pandas as pd

def load_processed_data(filepath):
    """Load processed data"""
    return pd.read_excel(filepath)

def person_based_split(df, n_train):
    """Split data by number of train participants"""
    participants = df['Participant'].unique()
    train_participants = participants[:n_train]
    test_participants = participants[n_train:]
    
    train_df = df[df['Participant'].isin(train_participants)]
    test_df = df[df['Participant'].isin(test_participants)]
    return train_df, test_df

def save_splits(train_df, test_df, output_dir):
    """Save train and test sets"""
    train_df.to_excel(f"{output_dir}/train.xlsx", index=False)
    test_df.to_excel(f"{output_dir}/test.xlsx", index=False)
    print(f"Train set: {len(train_df)} rows, {train_df['Participant'].nunique()} participants")
    print(f"Test set: {len(test_df)} rows, {test_df['Participant'].nunique()} participants")
    print(f"Files saved to {output_dir}")

if __name__ == "__main__":
    input_path = "../../data/processed/processed_20260219_000215.xlsx"
    output_dir = "../../data/processed"
    
    df = load_processed_data(input_path)
    total_participants = df['Participant'].nunique()
    
    print(f"Total participants: {total_participants}")
    n_train = int(input("How many for training? "))
    n_test = total_participants - n_train
    print(f"Test will have: {n_test} participants")
    
    train_df, test_df = person_based_split(df, n_train)
    save_splits(train_df, test_df, output_dir)
