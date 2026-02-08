#-----------------------------------------------------------------------
# starter.py
#
# Authors: Mui-Research Group @ ASDRP
# Date: 2026-02-07
# Description: Starter code for the LLM reasoning interpreter project
#-----------------------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# 1. SETUP: Load a small, capable model (e.g., Qwen-2.5-1.5B or Llama-3-8B)
model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def get_response_with_uncertainty(prompt):
    """
    Generates a response and calculates the average entropy (uncertainty).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with output_scores=True to get logits
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            output_scores=True, 
            return_dict_in_generate=True, 
            temperature=0.7, 
            do_sample=True
        )

    # 2. EXTRACT LOGITS & CALCULATE ENTROPY
    # Stack scores: (num_generated_tokens, batch_size, vocab_size)
    logits = torch.stack(outputs.scores, dim=0).squeeze(1) 
    probs = torch.softmax(logits, dim=-1)
    
    # Entropy formula: - sum(p * log(p))
    # We add 1e-9 to avoid log(0) errors
    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # Average entropy for the whole response (Simple metric)
    avg_entropy = torch.mean(entropy_per_token).item()
    
    # Decode answer
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer_text, avg_entropy

# 3. TEST ON A MATH PROBLEM
question = "Solve for x: 3x + 10 = 25"
prompt = f"User: {question}\nAssistant: Let's think step by step."

print(f"Question: {question}")
# Run 3 times to see 'Consistency'
for i in range(3):
    ans, ent = get_response_with_uncertainty(prompt)
    print(f"\nAttempt {i+1}:")
    print(f"Answer: {ans.strip()}")
    print(f"Uncertainty (Entropy): {ent:.4f}")

# QUESTIONS: 
# If 'Uncertainty' is high (> 1.5), is the answer more likely to be wrong?
# If the 3 attempts are different, is the answer more likely to be wrong?