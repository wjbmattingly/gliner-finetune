import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from .synthetic import process_example

def save_data(data, file_path, overwrite):
    """Save data to a file, handling overwriting based on user preference."""
    path = Path(file_path)
    assets_dir = path.parent
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
    
    if not overwrite and path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{path.stem}_{timestamp}{path.suffix}"
    
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Data saved to {file_path}")

def convert(data, project_path="", train_split=0.8, eval_split=0.2, test_split=0.0,
            train_file="train.json", eval_file="eval.json", test_file="test.json",
            overwrite=True):
    """Process data and split into training, validation, and testing datasets."""
    training_data = [process_example(example) for example in data]

    # Handle the data splitting
    if test_split > 0:
        train_val, test = train_test_split(training_data, test_size=test_split, random_state=42)
        save_data(test, Path(project_path, 'assets', test_file), overwrite)
    else:
        train_val = training_data

    eval_size = eval_split / (1 - test_split)  # Adjust eval size based on the remaining data
    train, val = train_test_split(train_val, test_size=eval_size, random_state=42)

    # Save the data
    save_data(train, Path(project_path, 'assets', train_file), overwrite)
    save_data(val, Path(project_path, 'assets', eval_file), overwrite)

    return training_data