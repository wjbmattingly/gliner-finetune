import json
import torch
from gliner import GLiNER
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os
from types import SimpleNamespace
from datetime import datetime

def extract_entity_types(data):
    """Extracts and aggregates all unique entity types from the dataset."""
    entity_types = set()
    for entry in data:
        if 'ner' in entry:
            for annotation in entry['ner']:
                entity_type = annotation[2]
                entity_types.add(entity_type)
    return entity_types

def load_json_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_model(model="urchade/gliner_small-v2.1"):
    try:
        return GLiNER.from_pretrained(model)
    except Exception as e:
        raise Exception(f"Failed to load model '{model}': {str(e)}")


def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types, 
        shuffle_types=config.shuffle_types, 
        random_drop=config.random_drop, 
        max_neg_type_ratio=config.max_neg_type_ratio, 
        max_len=config.max_len
    )
    
    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass
            
        # Check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:

            model.eval()
            
            if eval_data is not None:
                results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=0.5, batch_size=12,
                                     entity_types=eval_data["entity_types"])

                print(f"Step={step}\n{results}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)
                
            model.save_pretrained(f"{config.save_directory}/finetuned_{step}")

            model.train()

def train_model(model="urchade/gliner_small-v2.1", train_data="bird-training.json", 
                eval_data="bird-eval.json", num_steps=10, train_batch_size=10, 
                eval_every=10, project="project", device='cpu', lr_encoder=1e-5, 
                lr_others=5e-5, freeze_token_rep=False, max_types=25, shuffle_types=True, 
                random_drop=True, max_neg_type_ratio=1, max_len=384, overwrite_model=False):
    if project == "":
        project = os.getcwd()  # Fallback to current directory if no project path is provided

    save_directory = os.path.join(project, "outputs")

    # Handle model saving directory based on overwrite_model flag
    if not overwrite_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(save_directory, f"model_{timestamp}")
        os.makedirs(model_save_path, exist_ok=True)
    else:
        model_save_path = save_directory
    log_directory = os.path.join(model_save_path, "logs")
    os.makedirs(log_directory, exist_ok=True)
    config = SimpleNamespace(
        num_steps=num_steps,
        train_batch_size=train_batch_size,
        eval_every=eval_every,
        save_directory=log_directory,  # Ensure this is the intended directory for saving training logs
        warmup_ratio=0.1,
        device=device,
        lr_encoder=lr_encoder,
        lr_others=lr_others,
        freeze_token_rep=freeze_token_rep,
        max_types=max_types,
        shuffle_types=shuffle_types,
        random_drop=random_drop,
        max_neg_type_ratio=max_neg_type_ratio,
        max_len=max_len
    )

    try:
        model_instance = load_model(model)
        train_data = load_json_data(train_data)
        eval_data = load_json_data(eval_data)
        all_labels = extract_entity_types(eval_data)
        eval_data = {"entity_types": all_labels, "samples": eval_data}

        train(model_instance, config, train_data, eval_data)
        model_instance.save_pretrained(model_save_path, overwrite=overwrite_model)
    except Exception as e:
        print(f"Error during training: {e}")