# LLM Uncertainty Analysis: Quick Reference Guide

A companion document to "Understanding Reasoning and Uncertainty in Large Language Models"

---

## Quick Concepts Cheat Sheet

### Entropy Formula
```
H(P) = -Σ P(xᵢ) × log₂(P(xᵢ))
```
- **Low entropy (< 1.0)**: Confident, concentrated probability
- **High entropy (> 2.0)**: Uncertain, spread out probability

### Temperature Effect
```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```
- **T < 1**: Sharper distribution, more deterministic
- **T = 1**: Standard distribution
- **T > 1**: Flatter distribution, more random

### Self-Consistency Algorithm
1. Generate N samples with temperature > 0
2. Extract final answers
3. Count frequency of each answer
4. Select most common (majority vote)

---

## Entropy Interpretation Table

| Entropy Range | Confidence Level | Typical Behavior | Recommendation |
|---------------|------------------|------------------|----------------|
| 0.0 - 0.5 | Very High | Nearly deterministic | ✅ Trust |
| 0.5 - 1.0 | High | Confident, reliable | ✅ Usually safe |
| 1.0 - 1.5 | Moderate | Some uncertainty | ⚠️ Verify important facts |
| 1.5 - 2.5 | Low | Uncertain | ⚠️ Cross-check |
| > 2.5 | Very Low | Highly uncertain | ❌ Don't trust |

*Note: Ranges are approximate and model-dependent*

---

## Decision Matrix: When to Trust the Model

| Entropy | Consistency | Confidence | Action |
|---------|-------------|------------|--------|
| Low (<1.0) | High (3/3) | ✅ Very High | Trust the answer |
| Low (<1.0) | Medium (2/3) | ✅ High | Likely correct |
| Low (<1.0) | Low (1/3) | ⚠️ Suspicious | Investigate - possible systematic error |
| Medium (1.0-1.5) | High (3/3) | ✅ Good | Answer likely correct |
| Medium (1.0-1.5) | Medium (2/3) | ⚠️ Moderate | Verify if important |
| Medium (1.0-1.5) | Low (1/3) | ⚠️ Low | Don't rely on it |
| High (>1.5) | High (3/3) | ⚠️ Uncertain but Consistent | External verification needed |
| High (>1.5) | Medium (2/3) | ❌ Low | Unreliable |
| High (>1.5) | Low (1/3) | ❌ Very Low | Model is guessing |

---

## Code Snippets Library

### Basic Uncertainty Estimation

```python
def get_response_with_uncertainty(prompt, temperature=0.7, max_tokens=200):
    """Generate response and measure uncertainty via entropy."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=temperature,
            do_sample=True
        )
    
    # Calculate entropy
    logits = torch.stack(outputs.scores, dim=0).squeeze(1)
    probs = torch.softmax(logits, dim=-1)
    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    avg_entropy = torch.mean(entropy_per_token).item()
    
    # Decode answer
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer_text, avg_entropy
```

### Self-Consistency Check

```python
def check_self_consistency(prompt, num_samples=5, temperature=0.7):
    """Generate multiple samples and check consistency."""
    answers = []
    entropies = []
    
    for i in range(num_samples):
        answer, entropy = get_response_with_uncertainty(prompt, temperature)
        answers.append(answer.strip())
        entropies.append(entropy)
    
    # Count frequency of each answer
    from collections import Counter
    answer_counts = Counter(answers)
    most_common_answer, frequency = answer_counts.most_common(1)[0]
    
    consistency_ratio = frequency / num_samples
    avg_entropy = sum(entropies) / len(entropies)
    
    return {
        'most_common_answer': most_common_answer,
        'consistency_ratio': consistency_ratio,
        'all_answers': answers,
        'avg_entropy': avg_entropy,
        'entropy_std': np.std(entropies),
        'answer_distribution': dict(answer_counts)
    }
```

### Temperature Sweep

```python
def temperature_experiment(prompt, temps=[0.1, 0.5, 0.7, 1.0, 1.5, 2.0]):
    """Test different temperatures and analyze results."""
    results = []
    
    for temp in temps:
        answer, entropy = get_response_with_uncertainty(prompt, temperature=temp)
        results.append({
            'temperature': temp,
            'answer': answer.strip(),
            'entropy': entropy
        })
    
    return results
```

### Advanced: Per-Token Entropy Analysis

```python
def analyze_token_entropy(prompt, max_tokens=200):
    """Analyze entropy for each token to find uncertainty hotspots."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=0.7,
            do_sample=True
        )
    
    # Calculate per-token entropy
    logits = torch.stack(outputs.scores, dim=0).squeeze(1)
    probs = torch.softmax(logits, dim=-1)
    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # Get tokens
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    tokens = [tokenizer.decode([tok_id]) for tok_id in generated_ids]
    
    # Combine
    token_analysis = [
        {'token': tok, 'entropy': ent.item()}
        for tok, ent in zip(tokens, entropy_per_token)
    ]
    
    return token_analysis
```

### Confidence Classification

```python
def classify_confidence(entropy, consistency_ratio):
    """Classify model confidence based on metrics."""
    if entropy < 1.0 and consistency_ratio >= 0.8:
        return "VERY_HIGH"
    elif entropy < 1.5 and consistency_ratio >= 0.6:
        return "HIGH"
    elif entropy < 2.0 and consistency_ratio >= 0.4:
        return "MODERATE"
    elif entropy < 2.5 or consistency_ratio >= 0.4:
        return "LOW"
    else:
        return "VERY_LOW"
```

---

## Practical Experiment Templates

### Experiment 1: Benchmark Different Questions

```python
test_cases = {
    "trivial": "What is 1 + 1?",
    "easy": "What is the capital of France?",
    "medium": "Solve for x: 3x + 10 = 25",
    "hard": "What is the integral of x^2 from 0 to 5?",
    "very_hard": "Explain the Riemann hypothesis in simple terms.",
    "trick": "How many months have 28 days?",  # Trick: all of them
    "impossible": "What color is happiness?"  # Subjective/nonsensical
}

for difficulty, question in test_cases.items():
    prompt = f"User: {question}\nAssistant: Let's think step by step."
    result = check_self_consistency(prompt, num_samples=5)
    
    print(f"\n{difficulty.upper()}: {question}")
    print(f"Answer: {result['most_common_answer'][:100]}...")
    print(f"Consistency: {result['consistency_ratio']:.1%}")
    print(f"Avg Entropy: {result['avg_entropy']:.3f}")
    print(f"Confidence: {classify_confidence(result['avg_entropy'], result['consistency_ratio'])}")
```

### Experiment 2: Hallucination Detection

```python
# Questions designed to trigger hallucinations
hallucination_tests = [
    "Who won the 2025 Nobel Prize in Physics?",  # Future event
    "What is the capital of Atlantis?",  # Fictional place
    "When did Shakespeare meet Einstein?",  # Impossible
    "What does the word 'flibbertigibbet' mean in ancient Sumerian?",  # Made-up
]

for question in hallucination_tests:
    prompt = f"User: {question}\nAssistant:"
    result = check_self_consistency(prompt, num_samples=5)
    
    print(f"\nQuestion: {question}")
    print(f"Entropy: {result['avg_entropy']:.3f}")
    print(f"Consistency: {result['consistency_ratio']:.1%}")
    print(f"Answers: {result['answer_distribution']}")
    
    # High entropy + low consistency = likely hallucination
    if result['avg_entropy'] > 1.5 and result['consistency_ratio'] < 0.5:
        print("⚠️  WARNING: Likely hallucination detected!")
```

### Experiment 3: Optimal Sample Size

```python
def find_optimal_samples(prompt, max_samples=20):
    """Determine how many samples needed for stable estimates."""
    all_answers = []
    entropies = []
    
    # Generate all samples
    for i in range(max_samples):
        answer, entropy = get_response_with_uncertainty(prompt)
        all_answers.append(answer.strip())
        entropies.append(entropy)
    
    # Check stability at different sample sizes
    results = []
    for n in [1, 3, 5, 10, 15, 20]:
        sample_answers = all_answers[:n]
        sample_entropies = entropies[:n]
        
        from collections import Counter
        counts = Counter(sample_answers)
        majority, freq = counts.most_common(1)[0]
        
        results.append({
            'n_samples': n,
            'majority_answer': majority,
            'consistency': freq / n,
            'avg_entropy': np.mean(sample_entropies),
            'entropy_std': np.std(sample_entropies)
        })
    
    return results
```

---

## Troubleshooting Guide

### Problem: Entropy is always very high (> 3.0)

**Possible Causes:**
- Temperature too high → Try T=0.7 instead of 1.5+
- Model too small for task → Use larger model
- Prompt too vague → Add more context
- Task is genuinely very difficult

### Problem: All answers identical even with T=1.0

**Possible Causes:**
- Question too easy → Model very confident
- Temperature too low → Increase to 0.7-1.0
- Not using `do_sample=True` → Check generation params
- Greedy decoding activated → Ensure sampling enabled

### Problem: High entropy but correct answers

**Possible Causes:**
- Multiple valid phrasings → Expected behavior
- Creative task → High entropy normal
- Intermediate reasoning varied → Check final answer
- Need semantic entropy → Token-level misleading

### Problem: Low entropy but wrong answers

**Possible Causes:**
- Systematic bias in training → Model learned wrong pattern
- Trick question → Model missed nuance
- Out-of-distribution → Model overconfident on unfamiliar
- **This is the dangerous case!** → Why hallucination detection is hard

---

## Key Research Papers by Topic

### **If you're interested in... read these papers:**

**🎯 Getting Started:**
1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Wei et al. (2022) - "Chain-of-Thought Prompting"

**🔍 Uncertainty Basics:**
3. Wang et al. (2022) - "Self-Consistency Improves CoT"
4. Zhang et al. (2025) - "Survey of UQ Methods" (ACL)

**🚀 Advanced Methods:**
5. Farquhar et al. (2024) - "Semantic Entropy" (Nature)
6. Zhang et al. (2025) - "Entropy Turning Point for Temperature"

**💡 Practical Applications:**
7. Lou et al. (2024) - "Dynamic Temperature Sampling"
8. Xiong et al. (2025) - "Token-Entropy Conformal Prediction"

---

## Glossary

**Aleatoric Uncertainty**: Irreducible randomness in a system (e.g., coin flip).

**Chain-of-Thought (CoT)**: Prompting technique that encourages step-by-step reasoning.

**Entropy**: Mathematical measure of uncertainty; higher = more uncertain.

**Epistemic Uncertainty**: Uncertainty due to lack of knowledge (reducible with more info).

**Hallucination**: When an LLM generates plausible but incorrect information.

**Logits**: Raw scores before softmax; higher = more likely.

**Perplexity**: Measure of how "surprised" a model is by text; related to entropy.

**Sampling**: Randomly selecting next token from probability distribution.

**Self-Consistency**: Technique of generating multiple answers and using majority vote.

**Semantic Entropy**: Entropy measured over meanings rather than token sequences.

**Softmax**: Function converting scores to probabilities that sum to 1.

**Temperature**: Parameter controlling randomness; higher = more random.

**Token**: Basic unit of text (word, subword, or character).

**Uncertainty Quantification (UQ)**: Methods for measuring model reliability.

---

## Useful Resources

### Interactive Tools
- **Hugging Face Spaces**: Try models in browser
- **LM Studio**: Run models locally with GUI
- **OpenAI Playground**: Experiment with GPT models

### Learning Platforms
- **Hugging Face NLP Course**: Free, comprehensive
- **Fast.ai**: Practical deep learning
- **3Blue1Brown**: Visual explanations of concepts

### Code Libraries
- **Transformers**: Hugging Face library for LLMs
- **PyTorch**: Deep learning framework
- **LM-Polygraph**: UQ toolkit for LLMs

### Communities
- **r/MachineLearning**: Reddit community
- **Hugging Face Forums**: Technical discussions
- **Papers with Code**: Find implementations

---

## Citation Template

If you use this tutorial in your research or projects:

```
Tutorial: Understanding Reasoning and Uncertainty in Large Language Models
Project: Interpreter Research Project
Date: February 2026
URL: [Your repository URL]
```

---

**Quick Reference Version**: 1.0  
**Companion to**: understanding-llm-reasoning-uncertainty.md  
**Last Updated**: February 2026
