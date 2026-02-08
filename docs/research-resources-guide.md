# Research Resources Guide for High School Students

A curated guide to help high school researchers explore LLM uncertainty estimation and related topics.

---

## Table of Contents

1. [Getting Started with AI Research](#getting-started)
2. [Essential Papers (Annotated for Accessibility)](#essential-papers)
3. [Online Learning Resources](#online-resources)
4. [Tools and Platforms](#tools-and-platforms)
5. [How to Read Research Papers](#reading-papers)
6. [Project Ideas for High School Research](#project-ideas)
7. [Citing Sources Properly](#citations)
8. [Finding More Papers](#finding-papers)

---

## Getting Started with AI Research

### What Makes a Good Research Topic?

For high school level research in LLM uncertainty:

✅ **Good Topics:**
- Compare uncertainty across different model sizes
- Test uncertainty on specific domains (math, science, history)
- Visualize entropy patterns in model outputs
- Build a simple hallucination detector
- Analyze how temperature affects answer quality

❌ **Too Advanced (Save for College):**
- Novel theoretical frameworks
- Training new models from scratch
- Modifying model architectures
- Advanced mathematical proofs

### Research Skills You'll Develop

1. **Literature Review**: Finding and understanding prior work
2. **Experimental Design**: Setting up controlled tests
3. **Data Analysis**: Interpreting results statistically
4. **Scientific Writing**: Communicating findings clearly
5. **Critical Thinking**: Evaluating methods and results

---

## Essential Papers (Annotated for Accessibility)

### Tier 1: Start Here (Most Accessible)

#### 1. "Attention Is All You Need" (Vaswani et al., 2017)

**Why Read It:** Foundation of modern LLMs  
**Difficulty:** ⭐⭐⭐ (Moderate - has math but concepts are clear)  
**Key Sections to Focus On:**
- Section 1 (Introduction): Motivation for transformers
- Section 3.2 (Attention): The core mechanism
- Figures 1-2: Architecture diagrams

**What You Can Skip:**
- Detailed math in Section 3
- Training details in Section 5

**Link:** https://arxiv.org/abs/1706.03762

**Key Takeaway:** Transformers use attention to process sequences, replacing older RNN architectures.

---

#### 2. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)

**Why Read It:** Shows how to make LLMs show their work  
**Difficulty:** ⭐⭐ (Easy - minimal math, clear examples)  
**Key Sections:**
- Section 1-2: What is CoT and why it works
- Section 3: Experimental results (look at tables!)
- Figures: Examples of CoT vs. standard prompting

**Perfect for High School Because:**
- Lots of concrete examples
- Clear experimental setup you could replicate
- Minimal technical prerequisites

**Link:** https://arxiv.org/abs/2201.11903

**Key Takeaway:** Adding "Let's think step by step" dramatically improves reasoning accuracy.

---

#### 3. "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)

**Why Read It:** The technique your starter code implements!  
**Difficulty:** ⭐⭐ (Easy - very clear, good figures)  
**What to Focus On:**
- Algorithm 1: The self-consistency method
- Table 1: Performance improvements
- Figure 1: Visual explanation

**Great for Replication Study:**
- Simple to implement (you already have code!)
- Clear metrics to compare
- Easy to test on new problems

**Link:** https://arxiv.org/abs/2203.11171

**Key Takeaway:** Multiple reasoning paths to same answer = high confidence.

---

### Tier 2: Build Understanding (Moderate Difficulty)

#### 4. "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Farquhar et al., 2024)

**Why Read It:** Published in *Nature* - high-impact method  
**Difficulty:** ⭐⭐⭐ (Moderate - some statistics)  
**Accessible Parts:**
- Introduction and motivation (very clear)
- Figure 1: Main concept visualization
- Results sections (lots of graphs)

**Skip on First Read:**
- Detailed mathematical derivations
- Bayesian inference details

**Link:** https://doi.org/10.1038/s41586-024-07421-0  
**Free Version:** https://arxiv.org/abs/2406.15927

**Key Takeaway:** Group semantically similar answers before measuring entropy.

---

#### 5. "A Survey of Uncertainty Estimation Methods on Large Language Models" (Zhang et al., ACL 2025)

**Why Read It:** Comprehensive overview of ALL methods  
**Difficulty:** ⭐⭐⭐ (Moderate - survey style, good organization)  
**How to Use It:**
- Section 1-2: Motivation and problem setup
- Section 3+: Each subsection covers one method
- Tables: Compare different approaches

**Great as Reference:**
- Don't read cover-to-cover
- Use as encyclopedia when researching specific methods
- Look at comparison tables

**Link:** https://aclanthology.org/2025.findings-acl.1101/

**Key Takeaway:** Four main categories of uncertainty methods, each with trade-offs.

---

### Tier 3: Advanced Topics (For Ambitious Students)

#### 6. "Optimizing Temperature for Language Models with Multi-Sample Inference" (Zhang et al., 2025)

**Why Read It:** Recent (2025), practical technique  
**Difficulty:** ⭐⭐⭐⭐ (Challenging - statistics and algorithms)  
**What's Accessible:**
- Problem motivation
- Figures showing entropy patterns
- Experimental results

**Research Opportunity:**
- Could test their "entropy turning point" on different models
- Compare to fixed temperature
- Good science fair project!

**Link:** https://arxiv.org/abs/2502.05234

**Key Takeaway:** Optimal temperature correlates with entropy inflection point.

---

## Online Learning Resources

### Video Courses (Free)

#### 1. **3Blue1Brown: Neural Networks Series**
- **Link:** https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Time:** ~1.5 hours total
- **Why:** Best visual explanations of neural networks
- **Start with:** "But what is a neural network?"

#### 2. **3Blue1Brown: "But What Is a GPT?"**
- **Link:** https://www.youtube.com/watch?v=wjZofJX0v4M
- **Time:** 27 minutes
- **Why:** Visual explanation of transformers and attention
- **Perfect for:** Understanding before reading papers

#### 3. **StatQuest: Entropy (Information Theory)**
- **Link:** https://www.youtube.com/watch?v=YtebGVx-Fxw
- **Time:** 17 minutes
- **Why:** Clear explanation with examples
- **Great for:** Understanding entropy math

---

### Interactive Tutorials

#### 1. **The Illustrated Transformer** (Jay Alammar)
- **Link:** http://jalammar.github.io/illustrated-transformer/
- **Format:** Scrolling visual article
- **Time:** 30 minutes
- **Why:** Best visual breakdown of transformer architecture

#### 2. **Hugging Face NLP Course**
- **Link:** https://huggingface.co/learn/nlp-course/
- **Format:** Interactive lessons with code
- **Time:** Self-paced, ~20 hours total
- **Free:** Yes!
- **Why:** Hands-on practice with real models
- **Chapters to Focus On:**
  - Chapter 1: Transformer models
  - Chapter 2: Using transformers
  - Chapter 6: Generation

#### 3. **LLM Visualization**
- **Link:** https://bbycroft.net/llm
- **Format:** Interactive 3D visualization
- **Time:** 15 minutes to explore
- **Why:** See exactly how LLMs process text
- **Great for:** Visual learners

---

### Written Guides

#### 1. **"Attention? Attention!"** (Lilian Weng)
- **Link:** https://lilianweng.github.io/posts/2018-06-24-attention/
- **Level:** Intermediate
- **Why:** Clear explanations with code examples

#### 2. **"What Is Temperature in NLP?"** (Hugging Face)
- **Link:** https://huggingface.co/blog/how-to-generate
- **Level:** Beginner-Intermediate
- **Why:** Practical guide to generation parameters

#### 3. **Stanford CS324: Large Language Models**
- **Link:** https://stanford-cs324.github.io/winter2022/
- **Level:** College but accessible
- **Why:** Lecture notes and readings freely available
- **Focus on:** Lectures 1-4 for foundations

---

## Tools and Platforms

### Running Models

#### 1. **Google Colab**
- **Link:** https://colab.research.google.com/
- **Cost:** Free tier available
- **GPU:** Free GPU access (limited hours)
- **Best for:** Running experiments without local hardware
- **Tip:** Save work frequently, sessions time out

#### 2. **Hugging Face Spaces**
- **Link:** https://huggingface.co/spaces
- **Cost:** Free
- **Best for:** Testing models without coding
- **Try:** Search "text generation" and experiment

#### 3. **LM Studio** (Local)
- **Link:** https://lmstudio.ai/
- **Cost:** Free
- **Best for:** Running models on your computer
- **Requirement:** Good CPU/GPU helpful but not required
- **Why:** Privacy, no internet needed

---

### Coding and Development

#### 1. **Transformers Library** (Hugging Face)
- **Link:** https://github.com/huggingface/transformers
- **Language:** Python
- **Install:** `pip install transformers`
- **Why:** Industry standard, easy to use

#### 2. **Jupyter Notebooks**
- **Link:** https://jupyter.org/
- **Why:** Interactive coding, great for experiments
- **Tip:** Can run in Google Colab

#### 3. **Weights & Biases**
- **Link:** https://wandb.ai/
- **Cost:** Free for students
- **Best for:** Tracking experiments
- **Why:** Visualize results, compare runs

---

## How to Read Research Papers

### The Three-Pass Method

#### **Pass 1: The Quick Scan (5-10 minutes)**

Read ONLY:
- Title and abstract
- Section headings
- Introduction (first page)
- Conclusion
- Figures and captions

**Goal:** Decide if paper is relevant

**Ask yourself:**
- What problem does this solve?
- What's the main contribution?
- Is this relevant to my project?

---

#### **Pass 2: Understand the Gist (30-60 minutes)**

Read MORE carefully:
- Introduction fully
- Related work (to understand context)
- Method section (skim math, focus on concepts)
- Results (study tables and figures)
- Conclusion

SKIP on second pass:
- Detailed mathematical proofs
- Appendices
- Dense technical details

**Goal:** Understand main ideas and results

**Take notes on:**
- Main method/technique
- Key results
- Limitations mentioned
- Interesting figures

---

#### **Pass 3: Deep Dive (2-4 hours)**

Only for most important papers!

Read EVERYTHING:
- Work through math step-by-step
- Understand all algorithms
- Study code if available
- Read citations for background

**Goal:** Could you re-implement this?

---

### Tips for High School Students

**1. Don't Understand Everything on First Read**
- It's normal! Papers are written for experts
- Focus on main ideas, not every detail
- Come back to hard parts later

**2. Use Multiple Resources**
- Watch video explanation first
- Then read paper
- Then read blog posts
- Each reinforces understanding

**3. Make Notes in Your Own Words**
- Paraphrase key ideas
- Draw diagrams
- Explain to a friend (or rubber duck!)

**4. Focus on Empirical Results**
- Tables and graphs are often most accessible
- What did they test? What happened?
- Skip complex theory, focus on experiments

**5. Look for "Ablation Studies"**
- These test one thing at a time
- Great model for your own experiments!

---

## Project Ideas for High School Research

### Beginner Projects (1-2 weeks)

#### 1. **Temperature Tuning Study**
**Question:** What's the optimal temperature for different types of questions?  
**Method:** 
- Test temperatures from 0.1 to 2.0
- Try math, factual, creative prompts
- Measure accuracy and entropy

**Deliverable:** Chart showing optimal temperature by task type

---

#### 2. **Model Size vs. Uncertainty**
**Question:** Do bigger models have lower uncertainty?  
**Method:**
- Compare 3 model sizes (e.g., 1B, 3B, 7B parameters)
- Same questions to all models
- Measure and compare entropy

**Deliverable:** Visualization of size vs. confidence relationship

---

#### 3. **Subject Domain Analysis**
**Question:** Is the model more confident in some subjects?  
**Method:**
- Create questions in: math, history, science, literature
- Measure entropy for each domain
- Compare consistency

**Deliverable:** Subject confidence ranking

---

### Intermediate Projects (3-4 weeks)

#### 4. **Hallucination Detection System**
**Question:** Can we build a detector for made-up answers?  
**Method:**
- Collect examples of hallucinations (future events, impossible facts)
- Implement detection using entropy + consistency
- Test accuracy

**Deliverable:** Working detector with accuracy report

---

#### 5. **Chain-of-Thought Effectiveness**
**Question:** When does CoT help most?  
**Method:**
- Test with/without CoT on different problem types
- Measure accuracy improvement
- Categorize which problems benefit

**Deliverable:** Guidelines for when to use CoT

---

#### 6. **Visualization Dashboard**
**Question:** How can we visualize uncertainty for users?  
**Method:**
- Build web interface showing entropy
- Color-code high/medium/low confidence
- Add consistency indicators

**Deliverable:** Interactive demo

---

### Advanced Projects (6-8 weeks, Science Fair Quality)

#### 7. **Semantic Clustering Study**
**Question:** Does semantic grouping improve hallucination detection?  
**Method:**
- Implement simple semantic clustering
- Compare to token-level entropy
- Test on benchmark datasets

**Deliverable:** Full research paper with statistical analysis

---

#### 8. **Multi-Model Ensemble**
**Question:** Does combining uncertainty from multiple models help?  
**Method:**
- Run same prompts on 3-5 different models
- Aggregate uncertainty signals
- Compare to single-model methods

**Deliverable:** Conference-style paper and poster

---

#### 9. **Real-World Application**
**Question:** Can uncertainty estimation improve [specific application]?  
**Ideas:**
- Tutoring chatbot that knows when to ask teacher
- Fact-checker that flags uncertain claims
- Code assistant that shows confidence

**Deliverable:** Working prototype + evaluation

---

## Citing Sources Properly

### Citation Formats

#### APA Style (Most Common for Science)

**For arXiv Papers:**
```
Author(s). (Year). Title. arXiv preprint. https://arxiv.org/abs/XXXX.XXXXX
```

**Example:**
```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., 
Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. 
arXiv preprint. https://arxiv.org/abs/1706.03762
```

**For Published Papers:**
```
Author(s). (Year). Title. Journal/Conference, Volume(Issue), pages.
```

**Example:**
```
Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations 
in large language models using semantic entropy. Nature, 630, 625-630. 
https://doi.org/10.1038/s41586-024-07421-0
```

---

#### MLA Style (Common for High School)

**Example:**
```
Vaswani, Ashish, et al. "Attention Is All You Need." arXiv, 2017, 
arxiv.org/abs/1706.03762.
```

---

### Citation Management Tools

#### **Zotero** (Recommended for High School)
- **Link:** https://www.zotero.org/
- **Cost:** Free
- **Why:** Easy to use, browser extension, auto-formats

#### **Paperpile** (Google Docs Integration)
- **Link:** https://paperpile.com/
- **Cost:** Free trial, then paid
- **Why:** Works seamlessly with Google Docs

#### **Mendeley**
- **Link:** https://www.mendeley.com/
- **Cost:** Free
- **Why:** PDF annotation + citation management

---

### How to Cite Code

**GitHub Repository:**
```
Author. (Year). Repository name [Software]. GitHub. https://github.com/user/repo
```

**Python Package:**
```
Author. (Year). Package name (Version X.X.X) [Software]. PyPI. https://pypi.org/project/name/
```

**Your Own Code:**
```
Available at: https://github.com/yourusername/yourproject
```

---

## Finding More Papers

### Search Engines for Papers

#### 1. **Google Scholar**
- **Link:** https://scholar.google.com/
- **Best for:** Finding papers and citations
- **Tip:** Use "cited by" to find newer related work

#### 2. **Semantic Scholar**
- **Link:** https://www.semanticscholar.org/
- **Best for:** AI/CS papers with better summaries
- **Feature:** Shows influential citations

#### 3. **arXiv**
- **Link:** https://arxiv.org/
- **Best for:** Newest papers (not yet peer-reviewed)
- **Browse:** cs.CL (Computation and Language) or cs.AI

#### 4. **Papers with Code**
- **Link:** https://paperswithcode.com/
- **Best for:** Papers with implementation code
- **Feature:** Leaderboards for benchmarks

---

### Search Strategies

**Basic Search:**
```
"language model" AND "uncertainty" AND "entropy"
```

**Find Recent Work:**
```
"language model uncertainty" after:2024
```

**Find Surveys:**
```
"survey" OR "review" "language model uncertainty"
```

**Find Tutorials:**
```
"tutorial" OR "introduction" "transformers"
```

---

### Staying Up to Date

#### **Twitter/X**
- Follow: @_akhaliq (daily arXiv papers)
- Follow: @hardmaru (ML researcher)
- Hashtag: #NLProc, #MachineLearning

#### **Newsletters**
- **Import AI** (Jack Clark): Weekly ML news
- **The Batch** (deeplearning.ai): Weekly roundup

#### **Reddit**
- r/MachineLearning: Research discussions
- r/LanguageTechnology: NLP focused

#### **YouTube Channels**
- **Yannic Kilcher**: Paper explanations
- **Two Minute Papers**: Quick paper summaries

---

## Tips for Success

### 1. Start Small, Dream Big
- Begin with simple experiments
- Build confidence with successes
- Gradually increase complexity

### 2. Document Everything
- Keep a research journal
- Note what works AND what doesn't
- Failed experiments are still learning!

### 3. Ask for Help
- Professors love curious students
- Online communities are helpful
- Don't struggle alone

### 4. Focus on Reproducibility
- Save all code and data
- Document your methods clearly
- Others should be able to repeat your work

### 5. Communicate Clearly
- Write for classmates, not experts
- Use visuals and examples
- Explain why your work matters

---

## Additional Resources

### Books (Accessible Level)

1. **"Grokking Deep Learning"** by Andrew Trask
   - Great for beginners
   - Build neural networks from scratch
   - High school friendly

2. **"Make Your Own Neural Network"** by Tariq Rashid
   - Very beginner-friendly
   - Minimal math prerequisites
   - Python code included

3. **"Deep Learning for Coders"** by Jeremy Howard & Sylvain Gugger
   - Practical, code-first approach
   - Free online: https://course.fast.ai/

---

### Online Communities

1. **Hugging Face Discord**
   - Link: https://discuss.huggingface.co/
   - Very beginner-friendly
   - Active community

2. **r/learnmachinelearning** (Reddit)
   - Beginner questions welcome
   - Lots of resources shared

3. **AI Stack Exchange**
   - Link: https://ai.stackexchange.com/
   - Q&A format
   - Good for specific questions

---

## Conclusion

Research in AI is exciting and accessible to high school students! You don't need a PhD to contribute meaningful insights. Start with the resources in this guide, work through the tutorials, and build something interesting.

**Remember:**
- Every expert was once a beginner
- Understanding comes with practice
- Your fresh perspective is valuable
- Questions lead to discoveries

**Good luck with your research! 🚀🔬🤖**

---

## Quick Reference: Best Starting Path

**Week 1:**
1. Watch 3Blue1Brown GPT video
2. Read "The Illustrated Transformer"
3. Run the starter code

**Week 2:**
4. Read Chain-of-Thought paper
5. Read Self-Consistency paper  
6. Try temperature experiments

**Week 3:**
7. Pick a project idea
8. Design experiments
9. Collect data

**Week 4:**
10. Analyze results
11. Make visualizations
12. Write up findings

**You've got this! 📚✨**

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Target Audience**: High School Researchers  
**Companion to**: understanding-llm-reasoning-uncertainty.md
