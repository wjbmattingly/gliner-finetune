# GLiNER-Finetune

`gliner-finetune` is a Python library designed to generate synthetic data using OpenAI's GPT models, process this data, and then use it to train a GLiNER model. GLiNER is a framework for learning and inference in Named Entity Recognition (NER) tasks.

## Features

- **Data Generation**: Leverage OpenAI's powerful language models to create synthetic training data.
- **Data Processing**: Convert raw synthetic data into a format suitable for NER training.
- **Model Training**: Fine-tune the GLiNER model on the processed synthetic data for improved NER performance.

## Installation

To install the `gliner-finetune` library, use pip:

```bash
pip install gliner-finetune
```

## Quick Start

The following example demonstrates how to generate synthetic data, process it, and train a GLiNER model using the `gliner-finetune` library.

### Step 1: Generate Synthetic Data

```python
from gliner_finetune.synthetic import generate_data, create_prompt
import json

# Define your example data
example_data = {
    "text": "The Alpine Swift primarily consumes flying insects such as wasps, bees, and flies. It captures its prey mid-air while swiftly flying through the alpine skies. It nests in high, rocky mountain crevices where it uses feathers and small sticks to construct a simple yet secure nesting environment.",
    "generic_plant_food": [],
    "generic_animal_food": ["flying insects"],
    "plant_food": [],
    "specific_animal_food": ["wasps", "bees", "flies"],
    "location_nest": ["rocky mountain crevices"],
    "item_nest": ["feathers", "small sticks"]
}

# Convert example data to JSON string
json_data = json.dumps(example_data)

# Generate prompt and synthetic data
prompt = create_prompt(json_data)
print(prompt)

# Generate synthetic data with specified number of API calls
num_calls = 3
results = generate_data(json_data, num_calls)
print(results)
```

### Step 2: Process and Split Data

```python
from gliner_finetune.convert import convert

# Assuming the data has been read from 'parsed_responses.json'
with open('synthetic_data/parsed_responses.json', 'r') as file:
    data = json.load(file)

# Flatten the data list for processing
final_data = [sample for item in data for sample in item]

# Convert and split the data into training, validation, and testing datasets
training_data = convert(final_data, project_path='', train_split=0.8, eval_split=0.2, test_split=0.0,
                        train_file='train.json', eval_file='eval.json', test_file='test.json', overwrite=True)
```

### Step 3: Train the GLiNER Model

```python
from gliner_finetune.train import train_model

# Train the model
train_model(model="urchade/gliner_small-v2.1", train_data="assets/train.json", 
            eval_data="assets/eval.json", project="")
```

## Documentation

For more details about the GLiNER model and its capabilities, visit the official repository:

- [GLiNER GitHub Repository](https://github.com/urchade/GLiNER)