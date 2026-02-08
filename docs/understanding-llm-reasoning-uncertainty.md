# Understanding Reasoning and Uncertainty in Large Language Models


## Table of Contents

1. [Introduction](#introduction)
2. [Why This Matters: The Problem of LLM Hallucinations](#why-this-matters)
3. [Historical Context: From Rule-Based AI to Neural Language Models](#historical-context)
4. [The Science of Uncertainty](#the-science-of-uncertainty)
5. [Understanding Entropy: Information Theory Fundamentals](#understanding-entropy)
6. [How LLMs Generate Text: From Probabilities to Words](#how-llms-generate-text)
7. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
8. [Self-Consistency and Multiple Sampling](#self-consistency)
9. [Code Walkthrough: Building an Uncertainty Analyzer](#code-walkthrough)
10. [Experimental Results and Interpretation](#experimental-results)
11. [Advanced Topics: Recent Research (2024-2025)](#advanced-topics)
12. [Limitations and Future Directions](#limitations)
13. [Hands-On Experiments](#hands-on-experiments)
14. [References and Further Reading](#references)

---

## Introduction

Imagine asking a brilliant student to solve a math problem. Sometimes they answer confidently and correctly. Other times, they hesitate, showing uncertainty. And occasionally, they answer confidently but incorrectly—they're "hallucinating" an answer.

Large Language Models (LLMs) like GPT-4, Claude, and Qwen face the same challenge. They can generate fluent, convincing text, but **we need to know when they're confident versus when they're guessing**. This tutorial explores how we can measure an LLM's uncertainty to understand when it's reasoning reliably.

**What You'll Learn:**
- How LLMs generate text using probability distributions
- What entropy means and why it measures uncertainty
- How to detect when an LLM might be hallucinating
- Practical Python code to analyze LLM reasoning
- The latest research techniques (2024-2025) in uncertainty quantification

**Prerequisites:**
- Basic Python programming
- High school mathematics (logarithms, probability basics)
- Curiosity about how AI works!

---

## Why This Matters: The Problem of LLM Hallucinations

### The Challenge

Large Language Models have revolutionized AI by demonstrating remarkable capabilities in writing, reasoning, and problem-solving. However, they have a critical flaw: **they can generate plausible-sounding but completely incorrect information**, a phenomenon called **hallucination**.

**Real-World Examples:**
- Medical chatbots suggesting incorrect treatments
- Legal AI citing non-existent court cases
- Students receiving confident but wrong answers to homework questions

### The Core Problem

LLMs don't "know" facts the way humans do. Instead, they predict the most likely next word based on patterns learned from vast amounts of text. This means:

1. **Confidence ≠ Correctness**: An LLM can be very confident about an incorrect answer
2. **No Built-in Verification**: The model doesn't check if its answer is true
3. **Hidden Uncertainty**: The model has internal uncertainty, but doesn't always express it

### Why Uncertainty Estimation Matters

According to recent research published in *Nature* (2024), uncertainty quantification is essential for:

- **Hallucination Detection**: Identifying when the model is making things up
- **Selective Generation**: Knowing when to trust the model versus seeking human input
- **Trustworthy AI**: Building systems safe enough for high-stakes domains like healthcare and law

> "LLMs can produce hallucinated, biased, or non-factual responses despite fluent presentation, making reliable uncertainty quantification essential for trustworthy AI deployment." — ACL 2025 Findings Survey

---

## Historical Context: From Rule-Based AI to Neural Language Models

### The Evolution of AI Reasoning

**1950s-1980s: Symbolic AI**
- Computers used explicit rules: "If X then Y"
- Expert systems encoded human knowledge
- **Limitation**: Couldn't handle ambiguity or learn from data

**1990s-2000s: Statistical Methods**
- Introduction of probability into AI
- Hidden Markov Models for sequence prediction
- **Breakthrough**: Systems could learn patterns from data

**2010s: Deep Learning Revolution**
- Neural networks with multiple layers
- Recurrent Neural Networks (RNNs) for sequences
- **Problem**: Still struggled with long-range dependencies

**2017: The Transformer Architecture**
- "Attention Is All You Need" paper (Vaswani et al., 2017)
- Introduced the attention mechanism
- **Impact**: Enabled truly powerful language models

**2018-Present: The LLM Era**
- GPT series (OpenAI), BERT (Google), and others
- Models with billions of parameters
- Emergence of reasoning capabilities

### Understanding Attention: The Core Innovation

The **attention mechanism** allows models to focus on relevant information while ignoring less important parts—like how humans skim text by focusing on keywords.

**Key Idea**: When generating the next word, the model "attends to" different parts of the input with different weights:

```
Input: "The cat sat on the ___"
Attention weights: "cat" (high), "sat" (medium), "on" (medium), "the" (low)
Output: "mat" (likely completion)
```

This enables transformers to:
- Understand context over long distances
- Perform multi-step reasoning
- Generate coherent, contextual responses

---

## The Science of Uncertainty

### What is Uncertainty?

In science and mathematics, **uncertainty** quantifies our lack of knowledge. There are two main types:

**1. Aleatoric Uncertainty (Randomness)**
- Inherent randomness in a system
- Example: Flipping a fair coin—you can't predict the outcome
- Cannot be reduced with more data

**2. Epistemic Uncertainty (Knowledge)**
- Uncertainty due to lack of information
- Example: Not knowing if it will rain tomorrow
- Can be reduced by gathering more information

### Uncertainty in Language Models

When an LLM generates text, it experiences both types:

**Aleatoric**: Multiple valid ways to express the same idea
- "The answer is 5" vs. "It's 5" vs. "5 is the solution"

**Epistemic**: Lack of knowledge about the correct answer
- When asked about obscure facts
- When reasoning about novel problems

**The Key Insight**: We can measure epistemic uncertainty to detect when the model doesn't really "know" the answer.

### Probability Distributions: The Foundation

LLMs work with **probability distributions**—mathematical descriptions of how likely different outcomes are.

**Example**: For the prompt "The capital of France is ___", the model might output:

```
Paris:     0.95 (95% probability)
Lyon:      0.02 (2% probability)
London:    0.01 (1% probability)
Berlin:    0.01 (1% probability)
...others: 0.01 (1% probability)
```

- **High confidence**: Probability concentrated on one answer (Paris)
- **Low confidence**: Probability spread across many answers
- **Uncertainty**: How spread out the distribution is

---

## Understanding Entropy: Information Theory Fundamentals

### What is Entropy?

**Entropy** is a concept from information theory (Claude Shannon, 1948) that measures the **average uncertainty** or **surprise** in a probability distribution.

**Intuitive Definition**: 
- Low entropy = predictable, certain, concentrated
- High entropy = unpredictable, uncertain, spread out

### The Mathematical Definition

For a discrete probability distribution P with outcomes x₁, x₂, ..., xₙ:

```
H(P) = -Σ P(xᵢ) × log₂(P(xᵢ))
```

Where:
- H(P) is the entropy
- P(xᵢ) is the probability of outcome i
- log₂ is the logarithm base 2 (measures in "bits")

### Why This Formula Makes Sense

The formula captures three key principles:

**1. Rare Events Are Surprising**
- log₂(P(xᵢ)) is negative (since P < 1)
- Small probabilities give large negative logs
- The negative sign makes entropy positive

**2. Weighted by Probability**
- Events that never happen (P=0) don't contribute
- Common events contribute more to average uncertainty

**3. Maximum for Uniform Distribution**
- When all outcomes are equally likely, entropy is highest
- When one outcome is certain, entropy is zero

### Examples

**Example 1: Fair Coin (Maximum Uncertainty)**
```
P(Heads) = 0.5, P(Tails) = 0.5
H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
H = -(0.5 × -1 + 0.5 × -1) = 1 bit
```

**Example 2: Biased Coin (Lower Uncertainty)**
```
P(Heads) = 0.9, P(Tails) = 0.1
H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1))
H ≈ 0.47 bits
```

**Example 3: Certain Outcome (No Uncertainty)**
```
P(Heads) = 1.0, P(Tails) = 0.0
H = -(1.0 × log₂(1.0)) = 0 bits
```

### Entropy in Language Models

For LLMs, we calculate entropy **at each token** (word or sub-word) during generation:

- **Low entropy token**: Model is confident (e.g., "The capital of France is ___" → "Paris")
- **High entropy token**: Model is uncertain (e.g., "The meaning of life is ___" → many possibilities)

**Average Entropy**: By averaging entropy across all generated tokens, we get a single metric for the model's overall uncertainty about its response.

---

## How LLMs Generate Text: From Probabilities to Words

### The Generation Process

When you prompt an LLM, here's what happens step-by-step:

**1. Tokenization**
```
Input: "Solve for x: 3x + 10 = 25"
Tokens: ["Sol", "ve", " for", " x", ":", " 3", "x", " +", " 10", " =", " 25"]
```

**2. Encoding**
- Tokens converted to numerical vectors (embeddings)
- Position information added

**3. Transformer Processing**
- Multiple attention layers process the input
- Each layer builds more abstract representations
- Final layer produces "logits" (raw scores) for every possible next token

**4. Softmax: Converting Scores to Probabilities**

The logits are converted to probabilities using the **softmax function**:

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

Where T is the **temperature** parameter (more on this later).

**5. Sampling**

Instead of always picking the highest probability token (greedy decoding), we often **sample** from the distribution:
- Introduces variety and creativity
- Multiple samples can reveal uncertainty

**6. Decoding**
- Selected token is added to the sequence
- Process repeats until done

### The Role of Temperature

**Temperature** (T) controls the "sharpness" of the probability distribution:

**Low Temperature (T < 1)**
```
Before (T=1):  [0.4, 0.3, 0.2, 0.1]
After (T=0.5): [0.53, 0.28, 0.14, 0.05]  # More peaked
```
- Makes distribution sharper
- More conservative, repetitive
- Less diverse outputs

**High Temperature (T > 1)**
```
Before (T=1):  [0.4, 0.3, 0.2, 0.1]
After (T=2.0): [0.32, 0.29, 0.24, 0.15]  # Flatter
```
- Makes distribution flatter
- More creative, diverse
- Higher entropy

**Optimal Temperature**: Recent research (2025) shows that the best temperature often aligns with the "entropy turning point"—where token-level entropy shifts from concave to convex behavior.

---

## Chain-of-Thought Reasoning

### What is Chain-of-Thought?

**Chain-of-Thought (CoT)** prompting encourages LLMs to break down complex problems into step-by-step reasoning, similar to showing your work in math class.

**Example Without CoT:**
```
Q: Solve for x: 3x + 10 = 25
A: x = 5
```

**Example With CoT:**
```
Q: Solve for x: 3x + 10 = 25
A: Let's think step by step.
1. We have 3x + 10 = 25
2. Subtract 10 from both sides: 3x = 15
3. Divide both sides by 3: x = 5
Therefore, x = 5
```

### Why Chain-of-Thought Works

**1. Breaks Down Complex Problems**
- Simpler intermediate steps
- Each step builds on the previous

**2. Mimics Human Reasoning**
- Trained on text showing step-by-step solutions
- Activates reasoning patterns learned during training

**3. Improves Accuracy**
- Fewer errors in multi-step problems
- Makes reasoning transparent and verifiable

**4. Enables Error Detection**
- We can check each step
- Model can "catch" its own mistakes

### The Science Behind It

Research shows that CoT prompting particularly helps with:
- **Arithmetic reasoning**: Multi-step calculations
- **Commonsense reasoning**: Logical deduction
- **Symbolic reasoning**: Abstract problem-solving

The improvement comes from transformers' ability to perform **approximate symbolic computation** through attention mechanisms, as explained by recent Vector Symbolic Architecture (VSA) interpretations of transformers.

---

## Self-Consistency and Multiple Sampling

### The Core Idea

**Self-Consistency** (Wang et al., 2022) is based on a simple but powerful intuition:

> "A complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer."

**Key Insight**: If the model is confident about the correct answer, **different reasoning paths should converge to the same result**. If the model is uncertain or wrong, different paths will give different answers.

### The Algorithm

1. **Sample Multiple Reasoning Paths**
   - Generate N different CoT responses (e.g., N = 5-10)
   - Use temperature > 0 for diversity

2. **Extract Final Answers**
   - Parse the final answer from each path
   - Ignore the intermediate reasoning steps

3. **Majority Vote**
   - Count how many times each answer appears
   - Select the most frequent answer

### Performance Improvements

Original self-consistency paper (2022) showed dramatic improvements:

- **GSM8K** (grade school math): +17.9% accuracy
- **SVAMP** (math word problems): +11.0% accuracy
- **AQuA** (algebraic questions): +12.2% accuracy
- **StrategyQA** (strategy questions): +6.4% accuracy
- **ARC-challenge** (science questions): +3.9% accuracy

### Recent Advances (2025)

**Confidence-Informed Self-Consistency (CISC)**:
- Uses **weighted voting** based on model confidence scores
- Reduces required samples by 40%+ while improving accuracy
- Better than simple frequency counting

**Reasoning-Pruning Perplexity Consistency (RPC)**:
- Combines perplexity with reasoning pruning
- Reduces sampling cost by 50%
- Exponential convergence of estimation error

### Detecting Uncertainty via Consistency

**Consistency as an Uncertainty Signal**:

```
High Confidence (Low Uncertainty):
Sample 1: "x = 5"
Sample 2: "x = 5"
Sample 3: "x = 5"
→ 100% agreement → High confidence

Medium Confidence (Medium Uncertainty):
Sample 1: "x = 5"
Sample 2: "x = 5"
Sample 3: "x = 4"
→ 67% agreement → Moderate confidence

Low Confidence (High Uncertainty):
Sample 1: "x = 5"
Sample 2: "x = 4"
Sample 3: "x = 3"
→ 33% agreement → Low confidence
```

**This is exactly what our code tests!**

---

## Code Walkthrough: Building an Uncertainty Analyzer

Now let's examine the Python code that implements these concepts. We'll break it down section by section.

### Setup and Model Loading

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load a small, capable model
model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```

**What's Happening:**

1. **Environment Variables**: Loads API keys or configuration from `.env`
2. **PyTorch**: Deep learning framework for running the model
3. **Transformers**: Hugging Face library providing easy access to pre-trained models
4. **Model Selection**: Qwen2.5-1.5B-Instruct is chosen because:
   - Small enough to run on consumer hardware (1.5 billion parameters)
   - Instruction-tuned for following prompts
   - Capable of basic reasoning tasks

**Device Mapping**: `device_map="auto"` automatically uses GPU if available, CPU otherwise.

### The Core Function: Uncertainty Estimation

```python
def get_response_with_uncertainty(prompt):
    """
    Generates a response and calculates the average entropy (uncertainty).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
```

**Tokenization:**
- Converts text to token IDs
- `return_tensors="pt"` returns PyTorch tensors
- `.to(model.device)` moves to GPU/CPU

```python
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
```

**Generation Parameters:**

- `torch.no_grad()`: Disables gradient computation (faster, uses less memory)
- `max_new_tokens=200`: Generate up to 200 tokens
- `output_scores=True`: **Critical!** Returns logits for each token
- `return_dict_in_generate=True`: Returns structured output
- `temperature=0.7`: Moderate randomness (between 0=deterministic and 1=standard)
- `do_sample=True`: Sample from distribution rather than greedy decoding

### Calculating Entropy

```python
    # Extract logits & calculate entropy
    logits = torch.stack(outputs.scores, dim=0).squeeze(1) 
    probs = torch.softmax(logits, dim=-1)
```

**Understanding the Tensors:**

- `outputs.scores`: List of logit tensors, one per generated token
- `torch.stack()`: Combines into shape `(num_tokens, vocab_size)`
- `torch.softmax()`: Converts logits to probabilities
  - Applies along `dim=-1` (vocabulary dimension)
  - Each row sums to 1.0

```python
    # Entropy formula: - sum(p * log(p))
    # Add 1e-9 to avoid log(0) errors
    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # Average entropy for the whole response
    avg_entropy = torch.mean(entropy_per_token).item()
```

**Entropy Calculation:**

1. `probs * torch.log(probs + 1e-9)`: Element-wise multiplication
   - `1e-9`: Tiny constant prevents `log(0)` = undefined
2. `torch.sum(..., dim=-1)`: Sum across vocabulary for each token
3. Negative sign: Makes entropy positive
4. `torch.mean()`: Average across all generated tokens
5. `.item()`: Converts PyTorch tensor to Python number

**Why Average Entropy?**
- Gives single metric for overall response uncertainty
- High average = uncertain throughout
- Low average = confident throughout

### Decoding the Answer

```python
    # Decode answer
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer_text, avg_entropy
```

**Extraction Logic:**

- `outputs.sequences[0]`: Full sequence including prompt
- `[inputs.input_ids.shape[1]:]`: Slice to get only generated tokens
- `tokenizer.decode()`: Convert token IDs back to text
- `skip_special_tokens=True`: Remove `<|endoftext|>` and similar

### Testing with Multiple Samples

```python
# Test on a math problem
question = "Solve for x: 3x + 10 = 25"
prompt = f"User: {question}\nAssistant: Let's think step by step."

print(f"Question: {question}")
# Run 3 times to see 'Consistency'
for i in range(3):
    ans, ent = get_response_with_uncertainty(prompt)
    print(f"\nAttempt {i+1}:")
    print(f"Answer: {ans.strip()}")
    print(f"Uncertainty (Entropy): {ent:.4f}")
```

**Experimental Design:**

1. **Chain-of-Thought Prompt**: "Let's think step by step" triggers reasoning
2. **Multiple Runs**: 3 samples to check consistency
3. **Metrics**: Both answer text and entropy value

**Key Questions:**
- High entropy (> 1.5) → wrong answer?
- Different answers across attempts → wrong answer?
- Low entropy + consistent answers → correct answer?

---

## Experimental Results and Interpretation

### What to Expect

When you run the code, you might see output like this:

**Example 1: High Confidence, Correct Answer**
```
Question: Solve for x: 3x + 10 = 25

Attempt 1:
Answer: First, I'll subtract 10 from both sides: 3x = 15. Then divide by 3: x = 5.
Uncertainty (Entropy): 0.8234

Attempt 2:
Answer: 3x + 10 = 25, so 3x = 25 - 10 = 15, therefore x = 15/3 = 5.
Uncertainty (Entropy): 0.7891

Attempt 3:
Answer: Subtracting 10 from both sides gives 3x = 15. Dividing by 3 gives x = 5.
Uncertainty (Entropy): 0.8102
```

**Analysis:**
- ✅ **Low entropy** (~0.8): Model is confident
- ✅ **Consistent answers**: All say x = 5
- ✅ **Correct**: The answer is indeed 5
- **Conclusion**: High confidence justified

**Example 2: Medium Confidence, Inconsistent**
```
Attempt 1:
Answer: Let me solve this carefully... 3x = 15, so x = 5.
Uncertainty (Entropy): 1.2341

Attempt 2:
Answer: Hmm, if 3x + 10 = 25, then 3x = 35... wait, that's not right. x = 5.
Uncertainty (Entropy): 1.6782

Attempt 3:
Answer: I think x = 5, but let me double-check...
Uncertainty (Entropy): 1.4521
```

**Analysis:**
- ⚠️ **Medium-high entropy** (~1.2-1.7): Model somewhat uncertain
- ⚠️ **Hesitation markers**: "Hmm", "wait", "I think"
- ✅ **Final answers consistent**: Still converges to x = 5
- **Conclusion**: Model less certain but still correct

**Example 3: High Uncertainty, Wrong Answer**
```
Attempt 1:
Answer: Well, 3x + 10 = 25 means x = 15/3 = 5... or maybe 8?
Uncertainty (Entropy): 2.1456

Attempt 2:
Answer: If we divide everything by 3 first... x + 3.33 = 8.33, so x ≈ 5.
Uncertainty (Entropy): 2.3891

Attempt 3:
Answer: 3x = 35, so x = 11.67 approximately.
Uncertainty (Entropy): 2.0234
```

**Analysis:**
- ❌ **High entropy** (>2.0): Model very uncertain
- ❌ **Inconsistent reasoning**: Different approaches
- ❌ **Wrong answers**: Includes 8, 11.67
- **Conclusion**: High uncertainty correlates with errors

### Interpreting Entropy Values

Based on empirical observations and recent research:

**Entropy Range Guidelines:**

| Entropy | Confidence | Typical Behavior |
|---------|-----------|------------------|
| 0.0 - 0.5 | Very High | Nearly deterministic, often correct |
| 0.5 - 1.0 | High | Confident, usually reliable |
| 1.0 - 1.5 | Moderate | Some uncertainty, check consistency |
| 1.5 - 2.5 | Low | Uncertain, likely unreliable |
| > 2.5 | Very Low | Highly uncertain, often wrong |

**Important Caveats:**
- These ranges are approximate and model-dependent
- Task difficulty affects typical entropy
- Creative tasks may have higher entropy even when "correct"

### Consistency Analysis

**Perfect Consistency (3/3 same):**
- Strong signal of confidence
- Usually indicates correct answer
- Exception: Systematic errors learned during training

**Partial Consistency (2/3 same):**
- Moderate confidence
- Main answer likely correct
- Verify with additional samples

**No Consistency (3 different):**
- Low confidence
- Answer likely unreliable
- Model is "guessing"

### Combining Metrics: A Decision Framework

**Decision Matrix:**

| Entropy | Consistency | Interpretation | Action |
|---------|------------|----------------|--------|
| Low | High | Confident & Correct | ✅ Trust |
| Low | Low | Confident but Inconsistent | ⚠️ Investigate |
| High | High | Uncertain but Agreeing | ⚠️ Verify externally |
| High | Low | Uncertain & Disagreeing | ❌ Don't trust |

---

## Advanced Topics: Recent Research (2024-2025)

### Beyond Simple Entropy: Semantic Entropy

**The Problem with Token-Level Entropy:**

Traditional entropy treats different phrasings as different:
- "The answer is 5"
- "x equals 5"
- "5 is the solution"

These are **semantically identical** but have different token sequences!

**Semantic Entropy Solution** (Farquhar et al., Nature 2024):

Instead of measuring entropy over token sequences, measure it over **meaning clusters**:

1. Generate multiple responses
2. Group semantically equivalent responses together
3. Calculate entropy over these semantic groups

```
Traditional Entropy:
- "5" (30%)
- "x=5" (25%)  
- "Five" (20%)
- "4" (15%)
- "6" (10%)
→ High entropy (5 different outputs)

Semantic Entropy:
- Group 1: "5" equivalents (75%)
- Group 2: "4" (15%)
- Group 3: "6" (10%)
→ Lower entropy (3 semantic meanings)
```

**Results:**
- Better hallucination detection
- More robust across different models
- Published in *Nature*, June 2024

**Practical Implementation:**

Recent work introduced **Semantic Entropy Probes (SEPs)** that estimate semantic entropy from hidden states of a *single* generation, reducing computational cost to nearly zero while maintaining accuracy.

### Efficient Uncertainty Methods

**Token-Entropy Conformal Prediction (TECP)** (2025):
- Uses token-level entropy without requiring model logits
- Provides formal coverage guarantees
- Works across different model architectures

**Entropy-Aligned Decoding (EPIC)** (2025):
- Explicitly regulates uncertainty at each generation step
- Aligns sampling distribution entropy to data uncertainty
- Improves quality in creative writing, summarization, and math reasoning

### Dynamic Temperature Selection

**Entropy Turning Point (EntP)** (2025):
- Automatically selects optimal temperature
- Identifies where token entropy shifts from concave to convex
- No task-specific tuning needed
- Works across models, tasks, and sizes

**Entropy-Based Dynamic Temperature (EDT)** (2024):
- Adjusts temperature dynamically per token
- Higher temperature when model is confident
- Lower temperature when uncertain
- Better quality-diversity balance

### Uncertainty Quantification Survey

A comprehensive ACL 2025 survey identified four major approaches:

**1. Logit-Based Methods:**
- Use raw model scores
- Fast, no extra computation
- Example: Max probability, entropy

**2. Sampling-Based Methods:**
- Generate multiple outputs
- Measure agreement/consistency
- Example: Self-consistency, semantic entropy

**3. Model-Based Methods:**
- Train auxiliary uncertainty estimators
- Learn to predict reliability
- Example: Calibrated confidence models

**4. Hybrid Approaches:**
- Combine multiple signals
- Better performance than any single method
- Example: Combining entropy + consistency + perplexity

**Key Finding:** Ensembling multiple uncertainty scores significantly improves performance, especially for long-form generation.

---

## Limitations and Future Directions

### Current Limitations

**1. Computational Cost**
- Multiple sampling requires N forward passes
- Expensive for large models
- Barrier to real-time applications

**2. Calibration Issues**
- Entropy ranges vary by model and task
- No universal threshold
- Requires task-specific tuning

**3. Distribution Shift**
- Methods sensitive to out-of-distribution inputs
- Performance degrades on unusual prompts
- Vulnerable to adversarial examples

**4. Semantic Understanding**
- Token-level entropy misses semantic equivalence
- Semantic entropy requires expensive clustering
- Trade-off between accuracy and efficiency

**5. Correlation vs. Causation**
- High uncertainty correlates with errors
- Doesn't explain *why* the model is wrong
- Limited interpretability

### Open Research Questions

**1. Efficient Semantic Entropy**
- How to approximate semantic clustering cheaply?
- Can we learn semantic groupings?

**2. Calibration-Free Methods**
- Universal uncertainty metrics?
- Adaptive thresholds?

**3. Mechanistic Interpretability**
- What circuits in the model cause uncertainty?
- Can we intervene to reduce hallucinations?

**4. Multi-Modal Uncertainty**
- How to measure uncertainty in vision-language models?
- Uncertainty in code generation?

**5. Uncertainty-Aware Training**
- Can we train models to be better calibrated?
- Reinforcement learning from uncertainty feedback?

### Future Directions

**Near-Term (1-2 years):**
- Efficient semantic entropy approximations deployed in production
- Dynamic temperature/sampling as default
- Uncertainty scores in user interfaces

**Medium-Term (3-5 years):**
- Models trained to output calibrated uncertainty
- Mechanistic understanding of hallucination circuits
- Formal verification for critical applications

**Long-Term (5+ years):**
- Provably reliable AI systems
- Uncertainty-aware reasoning architectures
- Human-AI collaboration based on complementary uncertainties

---

## Hands-On Experiments

### Experiment 1: Entropy vs. Task Difficulty

**Hypothesis:** Harder problems should have higher entropy.

**Tasks to Try:**
```python
easy = "What is 2 + 2?"
medium = "Solve for x: 3x + 10 = 25"
hard = "Solve for x: x^2 - 5x + 6 = 0"
very_hard = "Prove that the square root of 2 is irrational."
```

**Expected Results:**
- Easy: Entropy < 0.5
- Medium: Entropy ~0.8-1.2
- Hard: Entropy ~1.5-2.5
- Very Hard: Entropy > 2.5

**Analysis Questions:**
- Does entropy increase with difficulty?
- Are harder problems less consistent?
- At what entropy does accuracy drop significantly?

### Experiment 2: Temperature Sweep

**Hypothesis:** Temperature affects both entropy and answer quality.

**Code:**
```python
temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
for temp in temperatures:
    # Modify generation to use this temperature
    # Track entropy and answer correctness
```

**Expected Results:**
- Low temp (0.1-0.3): Low entropy, repetitive, often correct
- Medium temp (0.5-0.7): Balanced entropy and quality
- High temp (1.5-2.0): High entropy, diverse but less accurate

**Analysis Questions:**
- What's the optimal temperature for math problems?
- Does optimal temperature change with problem difficulty?
- Is there a sweet spot for entropy?

### Experiment 3: Sample Size for Consistency

**Hypothesis:** More samples give better confidence estimates.

**Code:**
```python
sample_sizes = [1, 3, 5, 10, 20]
for n in sample_sizes:
    # Generate n samples
    # Calculate majority vote accuracy
    # Calculate confidence from agreement rate
```

**Expected Results:**
- n=1: No consistency check possible
- n=3: Basic consistency, may miss outliers
- n=5-10: Good balance of cost and reliability
- n=20: Diminishing returns

**Analysis Questions:**
- How many samples needed for reliable estimates?
- Does it depend on task difficulty?
- Cost-benefit trade-off?

### Experiment 4: Failure Mode Analysis

**Hypothesis:** We can categorize failure modes by uncertainty patterns.

**Test Cases:**
```python
cases = {
    "correct_confident": "2 + 2",
    "correct_uncertain": "Explain quantum entanglement",
    "wrong_confident": "What's the capital of Pangaea?",  # Trick question
    "wrong_uncertain": "Solve: x^10 + x^9 + ... + 1 = 0"
}
```

**Classification:**
- Correct + Low Entropy = ✅ Reliable
- Correct + High Entropy = ⚠️ Lucky guess?
- Wrong + Low Entropy = ❌ Hallucination!
- Wrong + High Entropy = ⚠️ Known uncertainty

**Analysis Questions:**
- What percentage fall into each category?
- Can we detect hallucinations reliably?
- Are there systematic patterns?

---

## References and Further Reading

### Foundational Papers

1. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *NeurIPS 2017*.
   - Original transformer paper
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **Shannon, C. E. (1948)**. "A Mathematical Theory of Communication." *Bell System Technical Journal*.
   - Foundation of information theory and entropy
   - [http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

### Chain-of-Thought and Self-Consistency

3. **Wei, J., et al. (2022)**. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*.
   - Introduced CoT prompting
   - [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

4. **Wang, X., et al. (2022)**. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023*.
   - Original self-consistency method
   - [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)

5. **Li, Z., et al. (2025)**. "Confidence-Informed Self-Consistency for LLMs."
   - Weighted voting improvement
   - [https://arxiv.org/abs/2502.06233](https://arxiv.org/abs/2502.06233)

### Uncertainty and Hallucination Detection

6. **Farquhar, S., et al. (2024)**. "Detecting hallucinations in large language models using semantic entropy." *Nature*, 630, 625-630.
   - Semantic entropy method published in Nature
   - [https://doi.org/10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0)

7. **Zhang, H., et al. (2025)**. "A Survey of Uncertainty Estimation Methods on Large Language Models." *ACL 2025 Findings*.
   - Comprehensive overview of UQ methods
   - [https://aclanthology.org/2025.findings-acl.1101/](https://aclanthology.org/2025.findings-acl.1101/)

8. **Xiong, M., et al. (2025)**. "TECP: Token-Entropy Conformal Prediction for LLMs."
   - Logit-free uncertainty with formal guarantees
   - [https://arxiv.org/abs/2509.00461](https://arxiv.org/abs/2509.00461)

### Temperature and Sampling

9. **Zhang, Y., et al. (2025)**. "Optimizing Temperature for Language Models with Multi-Sample Inference."
   - Entropy turning point method
   - [https://arxiv.org/abs/2502.05234](https://arxiv.org/abs/2502.05234)

10. **Lou, R., et al. (2024)**. "EDT: Entropy-Based Dynamic Temperature Sampling for LLMs."
    - Dynamic temperature adjustment
    - [https://arxiv.org/abs/2403.14541](https://arxiv.org/abs/2403.14541)

### Advanced Topics

11. **Braverman, M., et al. (2020)**. "Calibration, Entropy Rates, and Memory in Language Models." *ICML 2020*.
    - Entropy drift and calibration issues
    - [https://proceedings.mlr.press/v119/braverman20a.html](https://proceedings.mlr.press/v119/braverman20a.html)

12. **Ghojogh, B., & Ghodsi, A. (2024)**. "Attention Mechanism, Transformers, BERT, and GPT: Tutorial and Survey."
    - Comprehensive transformer tutorial
    - [https://hal.science/hal-04637647](https://hal.science/hal-04637647)

### Accessible Introductions

13. **3Blue1Brown**. "But what is a GPT? Visual intro to transformers."
    - Excellent video explanation
    - [https://www.youtube.com/watch?v=wjZofJX0v4M](https://www.youtube.com/watch?v=wjZofJX0v4M)

14. **Hugging Face Course**. "Natural Language Processing with Transformers."
    - Free interactive course
    - [https://huggingface.co/learn/nlp-course/](https://huggingface.co/learn/nlp-course/)

15. **The Illustrated Transformer**. Jay Alammar.
    - Visual guide to transformer architecture
    - [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

---

## Conclusion

Understanding uncertainty in large language models is crucial for building trustworthy AI systems. Through this tutorial, we've explored:

✅ **The Problem**: LLMs can hallucinate confidently incorrect answers  
✅ **The Theory**: Entropy measures uncertainty in probability distributions  
✅ **The Methods**: CoT reasoning, self-consistency, and semantic entropy  
✅ **The Practice**: Python code to measure and analyze LLM uncertainty  
✅ **The Future**: Recent advances making uncertainty estimation more efficient and reliable  

**Key Takeaways:**

1. **Entropy is a powerful signal** for detecting when models are uncertain
2. **Consistency across samples** reveals model confidence
3. **Combining multiple metrics** (entropy + consistency + semantics) works best
4. **Research is rapidly advancing** with new methods emerging monthly
5. **There's still much to discover** about how LLMs reason and fail

**Next Steps:**

- Run the experiments in this tutorial
- Try different models and compare their uncertainty patterns
- Read the referenced papers that interest you
- Contribute to open-source UQ tools
- Think creatively about new uncertainty metrics

The field of LLM uncertainty quantification is young and exciting—there's room for high school researchers to make meaningful contributions. We hope this tutorial inspires you to explore further!

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Author**: LLM Interpreter Research Project @ Mui-Group ASDRP
