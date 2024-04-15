import spacy
import json
from sklearn.model_selection import train_test_split
import json
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

def generate_data(source_json, num_calls, folder="synthetic_data", raw_filename="raw_responses.json", 
                      json_filename="parsed_responses.json", overwrite_file=False, model="gpt-4"):
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API Key must be set in the environment variables.")
    
    client = openai.OpenAI(api_key=api_key)  # Initialize the OpenAI client
    results_list = []  # Store raw string data
    json_results_list = []  # Store converted JSON data

    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Determine filenames
    if not overwrite_file:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        raw_filename = f"{timestamp}_{raw_filename}"
        json_filename = f"{timestamp}_{json_filename}"

    raw_path = os.path.join(folder, raw_filename)
    json_path = os.path.join(folder, json_filename)

    for _ in range(num_calls):
        prompt = create_prompt(source_json)
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model  # Use the desired model
            )
            results_list.append(response.choices[0].message.content)
            
            # Attempt to convert string data to JSON and save
            try:
                json_data = json.loads(response.choices[0].message.content)
                json_results_list.append(json_data)
            except json.JSONDecodeError:
                print("Failed to convert response to JSON.")

            # Save the raw string results
            with open(raw_path, 'w') as f:
                json.dump(results_list, f, indent=2)
            
            # Save the JSON-parsed results
            with open(json_path, 'w') as f:
                json.dump(json_results_list, f, indent=2)

        except Exception as e:
            print(f"API call failed: {e}")
            break

    return results_list, json_results_list

def create_prompt(json_data):
    """
    Generate a prompt from a JSON string to guide the AI in generating a specific type of output.
    This function assumes the JSON string represents a structured description of an object.

    :param json_data: A JSON string containing structured data about an object.
    :return: A string that is a well-formed prompt for the AI.
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON data provided.")

    # Construct the initial part of the prompt
    prompt = "Given the following JSON data, generate a list of 5 different comprehensive texts. This should only return valid JSON as a list of dictionaries. Do not say anything else. Each description should mimic the structure of the original input:\n\n"

    # Append the JSON data as a string directly into the prompt
    prompt += "JSON Data:\n" + json.dumps(data, indent=2) + "\n"

    # Instruct the AI to generate 5 different examples based on the data
    prompt += "\nGenerate 5 different examples in JSON format that follow the structure and content of the provided data."

    return prompt

def load_data(training_data):
    with open(training_data, "r") as f:
        data = json.load(f)
    return data

def create_patterns(example):
    patterns = []
    for category, items in example.items():
        if isinstance(items, list) and category != 'text':
            for item in items:
                patterns.append({"label": category.upper(), "pattern": item})
    return patterns

def process_example(example):
    nlp = spacy.blank("en")
    patterns = create_patterns(example)
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # Process the text
    doc = nlp(example["text"])

    # Preparing the output format
    tokenized_text = [token.text for token in doc]
    ner = []
    for ent in doc.ents:
        start = ent.start
        end = ent.end-1  # Adjusting end index to be inclusive, not +1
        ner.append([start, end, ent.label_])

    return {"tokenized_text": tokenized_text, "ner": ner}