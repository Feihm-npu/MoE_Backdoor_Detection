import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Generate Poisoned Dataset")
    parser.add_argument("--generator_model", type=str, default="distilgpt2", help="Model used to generate triggers")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--output_dir", type=str, default="./data/poisoned_agnews")
    parser.add_argument("--poison_rate", type=float, default=0.05, help="Proportion of data to poison (0.05 = 5%)")
    parser.add_argument("--target_label", type=int, default=1, help="Target label for backdoor (1 = Sports)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

class DynamicTriggerGenerator:
    """
    Implements the trigger generation described in the paper:
    Using a LM to generate context-aware suffixes (triggers)[cite: 223].
    """
    def __init__(self, model_name, device):
        print(f"Loading generator model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, text_batch):
        # Use the first 15 words as prefix context [cite: 222]
        prefixes = [" ".join(text.split()[:15]) for text in text_batch]
        
        inputs = self.tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,    # Length of the trigger
                do_sample=True,       # Sampling for diversity
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Extract only the generated suffix part as the trigger
        triggers = [full[len(pref):] for full, pref in zip(decoded, prefixes)]
        return triggers

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Load Clean Dataset
    print(f"Loading {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name)
    train_data = dataset["train"]
    
    # 2. Split into Clean and Poison subsets
    total_len = len(train_data)
    poison_count = int(total_len * args.poison_rate)
    
    # Random shuffle and split
    train_data = train_data.shuffle(seed=args.seed)
    poison_indices = range(poison_count)
    clean_indices = range(poison_count, total_len)
    
    poison_subset = train_data.select(poison_indices)
    clean_subset = train_data.select(clean_indices)
    
    print(f"Total: {total_len} | Poison: {poison_count} | Clean: {len(clean_subset)}")
    
    # 3. Initialize Generator
    generator = DynamicTriggerGenerator(args.generator_model, args.device)
    
    # 4. Apply Poisoning
    def poison_batch(batch):
        original_texts = batch["text"]
        triggers = generator.generate(original_texts)
        
        new_texts = []
        new_labels = []
        
        for text, trigger in zip(original_texts, triggers):
            # Dynamic Sentence Attack: Append generated trigger to original text
            poisoned_text = text + " " + trigger
            new_texts.append(poisoned_text)
            new_labels.append(args.target_label) # Label Flip 
            
        return {"text": new_texts, "label": new_labels}

    print("Generating triggers and poisoning data (this may take a while)...")
    # Batch processing for speed
    poisoned_dataset = poison_subset.map(poison_batch, batched=True, batch_size=32)
    
    # 5. Merge and Save
    final_dataset = concatenate_datasets([clean_subset, poisoned_dataset])
    final_dataset = final_dataset.shuffle(seed=args.seed) # Shuffle to mix poison data
    
    print(f"Saving processed dataset to {args.output_dir}...")
    final_dataset.save_to_disk(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()