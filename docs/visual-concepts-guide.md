# Visual Concepts Guide: LLM Uncertainty Explained with Diagrams

A visual companion to the main tutorial, using diagrams and illustrations to explain complex concepts.

---

## Table of Contents

1. [How LLMs Generate Text: Visual Flow](#text-generation-flow)
2. [Entropy Visualizations](#entropy-visualizations)
3. [Temperature Effect Diagrams](#temperature-effects)
4. [Self-Consistency Process](#self-consistency-visual)
5. [Chain-of-Thought Reasoning Path](#cot-visual)
6. [Uncertainty Detection Pipeline](#uncertainty-pipeline)
7. [Model Architecture Overview](#architecture-overview)

---

## Text Generation Flow

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER PROMPT                                  │
│              "Solve for x: 3x + 10 = 25"                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOKENIZATION                                 │
│  ["Solve", " for", " x", ":", " 3", "x", " +", " 10", ...]    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 EMBEDDING VECTORS                               │
│  [0.23, -0.45, 0.67, ...] [0.12, -0.34, ...]  ...             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMER LAYERS (×N)                            │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │ Attention  │ -> │  FFN       │ -> │ Layer      │  (repeat) │
│  │ Mechanism  │    │  Network   │    │ Norm       │           │
│  └────────────┘    └────────────┘    └────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT LOGITS                                  │
│  "Let": 8.2  "First": 7.9  "The": 7.1  "I": 5.3  ...         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           SOFTMAX (with Temperature)                            │
│  "Let": 0.35  "First": 0.28  "The": 0.18  "I": 0.05  ...     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAMPLING                                     │
│  Random selection based on probabilities → "Let"                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                REPEAT UNTIL DONE                                │
│  "Let" → "'s" → " think" → " step" → " by" → " step" ...      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Entropy Visualizations

### Probability Distributions and Entropy

#### Low Entropy (High Confidence)

```
Probability Distribution:
    █████████████████████ 95%  "Paris"
    █ 2%                       "Lyon"
    █ 1%                       "London"
    █ 1%                       "Berlin"
    █ 1%                       "Other"

Entropy ≈ 0.3 bits
Interpretation: Model is VERY CONFIDENT
Visualization: Tall, narrow spike
```

#### Medium Entropy (Moderate Confidence)

```
Probability Distribution:
    ████████████ 50%           "5"
    ████████ 35%               "five"
    ███ 10%                    "4"
    █ 5%                       "6"

Entropy ≈ 1.2 bits
Interpretation: Model is SOMEWHAT CONFIDENT
Visualization: Moderate spread
```

#### High Entropy (Low Confidence)

```
Probability Distribution:
    ████ 20%                   "happiness"
    ████ 18%                   "yellow"
    ███ 15%                    "blue"
    ███ 15%                    "warm"
    ███ 12%                    "bright"
    ███ 10%                    "undefined"
    ██ 10%                     "other"

Entropy ≈ 2.5 bits
Interpretation: Model is VERY UNCERTAIN
Visualization: Flat, spread out
```

### Entropy Calculation Visualization

```
For each outcome i:
    P(i) × log₂(P(i))
    
Example: Fair coin
┌────────┬──────────┬────────────┬──────────────┐
│ Outcome│   P(i)   │  log₂(P(i))│ P(i)×log₂(P(i))│
├────────┼──────────┼────────────┼──────────────┤
│ Heads  │   0.5    │    -1.0    │    -0.5      │
│ Tails  │   0.5    │    -1.0    │    -0.5      │
└────────┴──────────┴────────────┴──────────────┘
                           Sum: -1.0
                      Entropy: 1.0 bit

Example: Biased coin  
┌────────┬──────────┬────────────┬──────────────┐
│ Outcome│   P(i)   │  log₂(P(i))│ P(i)×log₂(P(i))│
├────────┼──────────┼────────────┼──────────────┤
│ Heads  │   0.9    │   -0.152   │   -0.137     │
│ Tails  │   0.1    │   -3.322   │   -0.332     │
└────────┴──────────┴────────────┴──────────────┘
                           Sum: -0.469
                     Entropy: 0.47 bits
```

---

## Temperature Effects

### Temperature Transformation

```
Original Logits (T=1.0):
Token     Logit    Probability
  A        2.0        0.40
  B        1.5        0.30
  C        1.0        0.20
  D        0.5        0.10

▼ Apply Temperature 0.5 (sharper)

After T=0.5:
Token   Logit÷0.5   Probability
  A        4.0         0.53  ████████████████████
  B        3.0         0.28  ██████████
  C        2.0         0.14  █████
  D        1.0         0.05  ██

Result: MORE CONFIDENT, LESS DIVERSE

▼ Apply Temperature 2.0 (flatter)

After T=2.0:
Token   Logit÷2.0   Probability
  A        1.0         0.32  ████████████
  B        0.75        0.29  ███████████
  C        0.5         0.24  █████████
  D        0.25        0.15  ██████

Result: LESS CONFIDENT, MORE DIVERSE
```

### Temperature Landscape

```
Temperature Scale:
0.0 ──────── 0.5 ──────── 1.0 ──────── 1.5 ──────── 2.0 ──────── ∞
│             │            │            │            │             │
Deterministic  Very Sharp  Standard    Smooth      Flat        Uniform
│             │            │            │            │             │
Always same   Repetitive  Balanced    Creative    Random      Chaos
│             │            │            │            │             │
No diversity  Low         Moderate    High        Very High   Maximum


Recommended Ranges by Task:
┌──────────────────────┬─────────────┬──────────────────┐
│ Task Type            │ Temperature │ Reasoning        │
├──────────────────────┼─────────────┼──────────────────┤
│ Math/Logic           │  0.3 - 0.7  │ Need precision   │
│ Factual Q&A          │  0.5 - 0.8  │ Prefer accuracy  │
│ Creative Writing     │  0.9 - 1.5  │ Want variety     │
│ Brainstorming        │  1.2 - 2.0  │ Maximum diversity│
│ Code Generation      │  0.2 - 0.6  │ Syntax sensitive │
└──────────────────────┴─────────────┴──────────────────┘
```

---

## Self-Consistency Visual

### The Process

```
                      ORIGINAL PROMPT
                            │
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   [Sample 1]          [Sample 2]          [Sample 3]
   T=0.7, seed=1       T=0.7, seed=2       T=0.7, seed=3
        │                   │                   │
        ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│"First subtract │  │"We start with  │  │"Let me solve   │
│ 10 from both   │  │ the equation   │  │ step by step:  │
│ sides: 3x = 15 │  │ 3x+10=25.      │  │ 3x+10=25       │
│ Then divide by │  │ Subtract 10:   │  │ 3x=25-10=15    │
│ 3: x = 5"      │  │ 3x=15          │  │ x=15÷3=5"      │
│                │  │ Divide by 3:   │  │                │
│ Entropy: 0.82  │  │ x=5"           │  │ Entropy: 0.91  │
│                │  │                │  │                │
│                │  │ Entropy: 0.76  │  │                │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         │    Extract Final Answer               │
         │                   │                   │
         ▼                   ▼                   ▼
        x=5                 x=5                 x=5
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                      ▼
              ┌───────────────────┐
              │  MAJORITY VOTE    │
              │                   │
              │  x=5: ███ (3/3)  │
              │  Consistency: 100%│
              │  Avg Entropy: 0.83│
              │                   │
              │  ✓ HIGH CONFIDENCE│
              └───────────────────┘
```

### Consistency Scenarios

```
Scenario A: PERFECT AGREEMENT (High Confidence)
────────────────────────────────────────────────
Sample 1: "x = 5"  ───┐
Sample 2: "x = 5"  ───┼──→  Majority: "x = 5" (100%)
Sample 3: "x = 5"  ───┘     ✓ Trust this answer


Scenario B: MAJORITY (Medium Confidence)
────────────────────────────────────────────────
Sample 1: "x = 5"  ───┐
Sample 2: "x = 5"  ───┼──→  Majority: "x = 5" (67%)
Sample 3: "x = 4"  ───┘     ⚠ Probably correct, verify


Scenario C: NO CONSENSUS (Low Confidence)
────────────────────────────────────────────────
Sample 1: "x = 5"  ───┐
Sample 2: "x = 4"  ───┼──→  No majority (33% each)
Sample 3: "x = 3"  ───┘     ✗ Don't trust, uncertain
```

---

## Chain-of-Thought Visual

### Without vs. With CoT

```
WITHOUT CHAIN-OF-THOUGHT:
─────────────────────────────────────────
Input:  "Solve: 3x + 10 = 25"
         │
         ▼
Output: "x = 5"

Problem: 
- No visibility into reasoning
- Can't check intermediate steps
- Higher error rate on complex problems


WITH CHAIN-OF-THOUGHT:
─────────────────────────────────────────
Input:  "Solve: 3x + 10 = 25
         Let's think step by step."
         │
         ▼
Output: "Step 1: We have 3x + 10 = 25
         Step 2: Subtract 10 from both sides
                 3x = 25 - 10 = 15
         Step 3: Divide both sides by 3
                 x = 15 ÷ 3 = 5
         Therefore, x = 5"

Benefits:
✓ Can verify each step
✓ Transparent reasoning
✓ Easier to catch errors
✓ Better accuracy
```

### Reasoning Path Diagram

```
Complex Problem: "John has 3 times as many apples as Mary.
                  Together they have 28 apples.
                  How many does Mary have?"

CoT REASONING PATH:
┌──────────────────────────────────────────────────────────┐
│ 1. UNDERSTAND                                            │
│    John = 3 × Mary                                       │
│    John + Mary = 28                                      │
│    Find: Mary = ?                                        │
└─────────┬────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│ 2. SUBSTITUTE                                            │
│    (3 × Mary) + Mary = 28                               │
│    4 × Mary = 28                                        │
└─────────┬────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│ 3. SOLVE                                                 │
│    Mary = 28 ÷ 4                                        │
│    Mary = 7                                             │
└─────────┬────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│ 4. VERIFY                                                │
│    Mary = 7, so John = 3 × 7 = 21                       │
│    Total = 7 + 21 = 28 ✓                                │
└──────────────────────────────────────────────────────────┘

Each step can be checked for correctness!
```

---

## Uncertainty Pipeline

### Complete Detection System

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT QUESTION                         │
│              "What is the capital of France?"              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │   Generate N Samples        │
        │   (with do_sample=True)     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│ Token-Level   │           │ Response-Level   │
│ Entropy       │           │ Consistency      │
│               │           │                  │
│ Calculate H   │           │ Compare answers  │
│ per token     │           │ Count agreement  │
│               │           │                  │
│ Avg entropy:  │           │ 5/5 say "Paris"  │
│ 0.45          │           │ = 100% consistent│
└───────┬───────┘           └────────┬─────────┘
        │                            │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   COMBINE SIGNALS          │
        │                            │
        │ Entropy:     0.45 (LOW)   │
        │ Consistency: 100% (HIGH)  │
        │                            │
        │ → CONFIDENCE: VERY HIGH   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   DECISION                 │
        │                            │
        │ ✓ Trust the answer        │
        │ ✓ Show to user            │
        │ ✓ No warning needed       │
        └────────────────────────────┘
```

### Hallucination Detection Flow

```
Question: "Who won the 2030 Nobel Prize in Physics?"
(Future event - should trigger uncertainty)

SAMPLE GENERATION:
─────────────────────────────────────────
Sample 1: "I believe it was Dr. Smith..."     Entropy: 2.3
Sample 2: "The winner was Prof. Johnson..."   Entropy: 2.7
Sample 3: "It went to Dr. Chen..."           Entropy: 2.4
Sample 4: "I think Dr. Smith received it..." Entropy: 2.1
Sample 5: "The award went to Prof. Lee..."   Entropy: 2.6

ANALYSIS:
─────────────────────────────────────────
Avg Entropy: 2.42 (VERY HIGH) ⚠️
Consistency: Smith(2), Johnson(1), Chen(1), Lee(1) = 40% ⚠️

Uncertainty markers:
├─ "I believe", "I think" → linguistic hedging ⚠️
├─ Different names in each sample ⚠️
└─ High entropy per token ⚠️

DECISION:
─────────────────────────────────────────
❌ LIKELY HALLUCINATION DETECTED

Actions:
├─ ⚠️  Show uncertainty warning to user
├─ 🔍 Suggest external verification
└─ 📊 Log for quality monitoring
```

---

## Model Architecture Overview

### Transformer Architecture (Simplified)

```
                    INPUT TOKENS
                    [I, love, AI]
                         │
                         ▼
              ┌────────────────────┐
              │  Token Embedding   │
              │  + Position Info   │
              └──────────┬─────────┘
                         │
         ┌───────────────┴────────────────┐
         │   TRANSFORMER BLOCK × N        │
         │                                │
         │  ┌──────────────────────────┐ │
         │  │  Multi-Head Attention    │ │ ← Looks at all tokens
         │  │                          │ │
         │  │  Q: What to look for?    │ │
         │  │  K: What do I have?      │ │
         │  │  V: What to return?      │ │
         │  └────────┬─────────────────┘ │
         │           │                    │
         │           ▼                    │
         │  ┌──────────────────────────┐ │
         │  │  Feed Forward Network    │ │ ← Process individually
         │  │                          │ │
         │  │  Dense → ReLU → Dense    │ │
         │  └────────┬─────────────────┘ │
         │           │                    │
         │           ▼                    │
         │  ┌──────────────────────────┐ │
         │  │  Layer Normalization     │ │ ← Stabilize
         │  └────────┬─────────────────┘ │
         │           │                    │
         └───────────┴────────────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │   Output Layer         │
         │   (Logits for each     │
         │    token in vocab)     │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │    Softmax             │
         │    (Probabilities)     │
         └──────────┬─────────────┘
                    │
                    ▼
              NEXT TOKEN
```

### Attention Mechanism Visualization

```
Query: "What should I pay attention to?"
Key:   "What information do I have?"
Value: "What do I actually return?"

Example: Generating next word after "The cat sat on the"

Input sequence:  [The]  [cat]  [sat]  [on]  [the]  [___]
                   │      │      │      │      │
Attention weights: 0.05   0.35   0.20   0.15   0.25
                   │      │      │      │      │
                   ▼      ▼      ▼      ▼      ▼
Interpretation:  Context
                   │      │
                   │      └─→ Focus on "cat" (subject)
                   │      
                   └─→ "sat on the" suggests location
                   
Most likely next: "mat", "floor", "rug"

Probability distribution:
  mat:   0.30  ██████████
  floor: 0.25  ████████
  rug:   0.20  ██████
  table: 0.10  ███
  other: 0.15  █████

Entropy: ~1.8 bits (moderate uncertainty)
```

---

## Uncertainty Metrics Comparison

### Visual Comparison Table

```
┌──────────────────────────────────────────────────────────────────┐
│                  UNCERTAINTY METRICS                             │
├──────────────┬────────────┬────────────┬──────────────┬─────────┤
│ Metric       │ Compute    │ Accuracy   │ Interpretable│ Cost    │
│              │ Required   │            │              │         │
├──────────────┼────────────┼────────────┼──────────────┼─────────┤
│ Max Prob     │ ██         │ ███        │ █████        │ $       │
│              │ Low        │ Moderate   │ Very High    │ Cheap   │
├──────────────┼────────────┼────────────┼──────────────┼─────────┤
│ Entropy      │ ███        │ ████       │ ████         │ $       │
│              │ Moderate   │ Good       │ High         │ Cheap   │
├──────────────┼────────────┼────────────┼──────────────┼─────────┤
│ Self-        │ █████      │ █████      │ ████         │ $$$$$   │
│ Consistency  │ High (×N)  │ Very Good  │ High         │ Expensive│
├──────────────┼────────────┼────────────┼──────────────┼─────────┤
│ Semantic     │ █████      │ █████████  │ ███          │ $$$$$   │
│ Entropy      │ Very High  │ Excellent  │ Moderate     │ Very    │
│              │            │            │              │ Expensive│
├──────────────┼────────────┼────────────┼──────────────┼─────────┤
│ Ensemble     │ ████████   │ ██████████ │ ███          │ $$$$$$$$│
│ Methods      │ Extreme    │ Best       │ Moderate     │ Extremely│
│              │            │            │              │ Expensive│
└──────────────┴────────────┴────────────┴──────────────┴─────────┘

Recommendation by Use Case:
─────────────────────────────
Real-time chatbot:     → Max Prob or Entropy
Research/Analysis:     → Semantic Entropy
Production (important):→ Self-Consistency
Maximum accuracy:      → Ensemble
```

---

## Practical Example: Full Workflow

```
RESEARCH QUESTION: "Is the model confident about this math problem?"

Problem: "If 2x + 5 = 13, what is x?"

STEP 1: GENERATE WITH ENTROPY TRACKING
═════════════════════════════════════════
Prompt: "User: If 2x + 5 = 13, what is x?
         Assistant: Let's solve step by step."

Generation Process:
Token 1: "First" → Entropy: 1.2  [moderate - multiple ways to start]
Token 2: ","     → Entropy: 0.3  [low - punctuation predictable]
Token 3: "subtract" → Entropy: 0.9  [low-mod - clear next step]
Token 4: "5"     → Entropy: 0.4  [low - specific number]
...
Token N: "4"     → Entropy: 0.2  [very low - confident answer]

Average Entropy: 0.72 → LOW → High confidence


STEP 2: SELF-CONSISTENCY CHECK
═════════════════════════════════════════
Run 5 times:

Sample 1: "First, subtract 5: 2x=8. Then x=4"    Ent: 0.72
Sample 2: "2x+5=13, so 2x=8, thus x=4"          Ent: 0.68
Sample 3: "Subtract 5 from both sides: 2x=8..."  Ent: 0.79
Sample 4: "We get 2x=13-5=8, x=8/2=4"           Ent: 0.71
Sample 5: "x = (13-5)/2 = 8/2 = 4"              Ent: 0.65

All answers: x=4 (5/5) → 100% consistency!


STEP 3: DECISION
═════════════════════════════════════════
┌────────────────────────────────────┐
│ CONFIDENCE ASSESSMENT              │
├────────────────────────────────────┤
│ Average Entropy:     0.71 (LOW)   │
│ Consistency:         100%  (HIGH) │
│                                    │
│ → VERY HIGH CONFIDENCE            │
│                                    │
│ ✓ Trust this answer               │
│ ✓ Answer is correct: x = 4       │
└────────────────────────────────────┘
```

---

## Key Insights Summary

### The Uncertainty Landscape

```
                    UNCERTAINTY SPACE
    
    Low Entropy                    High Entropy
    High Consistency               Low Consistency
         │                              │
         │                              │
         ▼                              ▼
    
┌─────────────────┐            ┌──────────────────┐
│  ✓ TRUST        │            │  ✗ DON'T TRUST   │
│                 │            │                  │
│ Model knows     │            │ Model guessing   │
│ Likely correct  │            │ Likely wrong     │
│ Safe to use     │            │ Need human check │
└─────────────────┘            └──────────────────┘
         │                              │
         │                              │
         ▼                              ▼
    
┌─────────────────┐            ┌──────────────────┐
│ ⚠ INVESTIGATE  │            │ ⚠ UNCERTAIN BUT │
│                 │            │   CONSISTENT     │
│ Confident but   │            │                  │
│ inconsistent    │            │ Hedging language │
│ Possible error  │            │ External verify  │
└─────────────────┘            └──────────────────┘

         Low Entropy                 High Entropy
         Low Consistency             High Consistency
```

---

## Conclusion

**Remember the Key Visual Principles:**

1. **Entropy = Spread** of probability distribution
   - Narrow spike = low entropy = confident
   - Flat distribution = high entropy = uncertain

2. **Temperature = Smoothness** control
   - Low temp = sharper = more deterministic
   - High temp = flatter = more random

3. **Consistency = Agreement** across samples
   - All same = high confidence
   - All different = low confidence

4. **Chain-of-Thought = Visible** reasoning path
   - Can check each step
   - Catches errors earlier

**Use these visual models to build intuition!**

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Companion to**: understanding-llm-reasoning-uncertainty.md
