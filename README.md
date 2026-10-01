# ML + Deep Learning Fundamentals — Full Beginner → GenAI Edition

> A friendly 10-day path from NumPy and Pandas to neural networks, Transformers, and the ideas behind modern GenAI.

[![Edition](https://img.shields.io/badge/Edition-Full%20Beginner%20%E2%86%92%20GenAI-blue)](https://github.com/Ravikiran9988/ML-DL-Fundamentals)
[![Year](https://img.shields.io/badge/Year-2026-informational)](https://github.com/Ravikiran9988/ML-DL-Fundamentals)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

## Resources

### Main Book

**[ML + Deep Learning Fundamentals — Full Beginner → GenAI Edition](PDF/ML-DL-Fundamentals-Book.pdf)**  

**[📄 Read the Markdown edition](MD/ML-Deep-Learning-Fundamentals.md)**

The 170-page main book is the complete learning path: a 10-day ML/deep-learning course, GenAI bridges, Project Ladder, and Study Toolkit.

### Assignment Guide

**[Assignment Guide — Hints, Solutions & Acceptance Criteria](PDF/Assignment-Guide-with-Hints-and-Solutions.pdf)**

The separate assignment guide is the practical companion for the learning path. It provides structured assignments, expected outputs, acceptance criteria, hints, solutions, submission guidance, rubrics, and debugging exercises.

The guide is intended to be used alongside the main book: **learn the concept in the book → attempt the assignment → check the acceptance criteria → use hints when needed → compare with the reference solution.**

## Overview

**ML + Deep Learning Fundamentals — Full Beginner → GenAI Edition** is a self-study learning path that builds machine-learning and deep-learning fundamentals first, then connects those foundations to modern GenAI systems.

The material is organized as a **10-day intensive curriculum** with explanations, analogies, small worked examples, code, assignments, quick recaps, and checkpoints. It is designed so that each day builds on the previous concepts instead of jumping directly into GenAI frameworks.

**First edition · 2026 · 170 pages · 3 parts · 62 index terms**

## What You Will Learn

### Part I — The 10-Day Course

| Day | Topic | Core coverage |
|---|---|---|
| 1 | **NumPy + Pandas** | Arrays, shapes, indexing, vectorization, broadcasting, matrix multiplication, cosine similarity, data inspection and cleaning |
| 2 | **Linear Regression + Gradient Descent** | Linear models, loss, gradients, gradient descent, fitting from scratch, comparison with scikit-learn |
| 3 | **Overfitting, Regularization & Cross-Validation** | Train/validation behavior, polynomial models, Ridge, Lasso, and cross-validation |
| 4 | **Metrics, Trees, Random Forest & Boosting** | Classification metrics, confusion matrices, class imbalance, decision trees, random forests, boosting |
| 5 | **k-NN, k-Means, PCA, Embeddings + Telco Churn** | Similarity and neighbors, clustering, dimensionality reduction, embeddings, and a customer-churn mini project |
| 6 | **Neural Networks From Scratch** | Neurons, activations, loss, backpropagation, a hidden layer, gradient checking, and experiments |
| 7 | **PyTorch** | Tensors, modules, training loops, optimizers, learning rates, train/validation tracking, and saving model weights |
| 8 | **MNIST, Dropout, Early Stopping & CNN** | MLP vs CNN, image shapes, dropout, training/validation loss, early stopping, and model comparison |
| 9 | **Attention & Transformers** | Self-attention, Q/K/V, scaled dot-product attention, masking, attention weights, and Transformer blocks |
| 10 | **Capstone + Recap** | End-to-end project work, results, plots, explanation, documentation, and transition to GenAI |

### Part II — GenAI Bridges

This section connects Transformer fundamentals to the main building blocks used in modern GenAI applications:

**LLMs → GenAI → Prompt Engineering → Embeddings → Vector DBs → RAG → Fine-tuning → AI Agents → Tool Calling → MCP → LangChain → LangGraph → AI Evaluation → AI Security → FastAPI → Docker → LLMOps**

The book explains the missing links rather than treating these topics as isolated buzzwords:

- **How Transformers become LLMs** — tokenization, token IDs, embeddings, positional information, Transformer blocks, logits, probabilities, and next-token generation.
- **Embeddings → Vector DB → RAG** — represent meaning as vectors, retrieve relevant information, and ground generated answers in retrieved context.
- **Tools, Agents, and MCP** — structured tool calls, validation, stateful multi-step workflows, step limits, logs, and the Model Context Protocol.
- **GenAI Evaluation and Security** — evaluation sets, answer quality, retrieval quality, faithfulness, prompt injection, least privilege, tool safety, data leakage, logging, and controlled failure.
- **Beginner-to-GenAI Project Ladder** — ten progressive project stages from data cleaning to a production GenAI API.

## The 10-Day Roadmap

```text
Day 1   NumPy + Pandas
   ↓
Day 2   Regression + Gradient Descent
   ↓
Day 3   Overfitting + Regularization + Cross-Validation
   ↓
Day 4   Metrics + Trees + Class Imbalance
   ↓
Day 5   k-NN + k-Means + PCA + Embeddings + Telco Churn
   ↓
Day 6   Neural Networks From Scratch
   ↓
Day 7   PyTorch
   ↓
Day 8   MNIST + Dropout + Early Stopping + CNN
   ↓
Day 9   Attention + Transformers
   ↓
Day 10  Capstone + Recap
   ↓
GenAI Bridges
   ↓
LLMs → Embeddings → RAG → Tools → Agents → MCP → Production
```

## Prerequisites

You do not need advanced mathematics before starting.

### Python essentials

- Variables, numbers, strings, booleans
- Lists, tuples, dictionaries, and sets
- `if/else` and loops
- Functions and return values
- Imports and modules
- Basic classes and objects
- Exceptions and reading tracebacks
- `pip` and virtual environments

### Math essentials

- Arithmetic, percentages, and averages
- Basic algebra and functions
- Vectors and matrices
- Dot product and matrix multiplication intuition
- Mean and variance
- Basic probability
- Derivative and gradient intuition

The book explicitly postpones advanced calculus and formal proofs until they are needed.

## How to Use the Material

The recommended learning loop in the book is:

```text
Understand
   ↓
See an analogy
   ↓
Work a tiny example
   ↓
Code it
   ↓
Break it
   ↓
Fix it
   ↓
Explain it
   ↓
Check yourself
   ↓
Move on
```

Each day includes:

- **Big Picture** — understand the idea before the details.
- **Detailed notes** — explanations, worked examples, code, and assignments.
- **Quick Recap** — compact revision material.
- **Checkpoint** — five questions to test understanding.

The book describes “10 days” as an **intensive path, not a promise of mastery**. Extra revision time may be needed, especially for backpropagation, PyTorch, CNNs, attention, and Transformers.

## Assignments & Projects

The assignments are designed to turn the concepts into working code and documented results.

### Day 1

**Cosine Similarity Without Loops**
- Implement matrix cosine similarity for `A (n,d)` and `B (m,d)`.
- Return an `(n,m)` similarity matrix.
- Normalize rows and use matrix multiplication.
- Verify the result against a loop implementation.

**Titanic Cleaning**
- Inspect and clean the dataset.
- Handle missing values and non-numeric columns.
- Explain dropped columns.
- Save `titanic_clean.csv`.
- Record three data insights.

### Day 2

**Linear Regression From Scratch**
- Implement gradient descent with NumPy.
- Track decreasing loss.
- Plot the loss curve and fitted line.
- Compare the learned result with scikit-learn.
- Explain `dw` and `db` in plain language.

### Day 3

**Overfitting, Regularization & Cross-Validation**
- Compare polynomial degrees 1–15.
- Plot training and validation behavior.
- Identify underfitting and overfitting.
- Compare Ridge and Lasso.
- Use 5-fold cross-validation and document conclusions.

### Day 4

**Credit Card Fraud Detection**
- Use a stratified split.
- Fit scaling only on training data.
- Compare three models.
- Report **precision, recall, F1, and ROC-AUC**.
- Show confusion matrices.
- Explain why accuracy alone can be misleading for highly imbalanced fraud data.

### Day 5

**Mini Project 1 — Telco Customer Churn**
- Convert `TotalCharges` to numeric.
- Use a `Pipeline` / `ColumnTransformer` to avoid leakage.
- Compare Logistic Regression with at least one stronger model.
- Use 5-fold stratified cross-validation.
- Report recall, precision, F1, and ROC-AUC.
- Include feature importance and a business recommendation.

The day also teaches **k-NN, k-Means, PCA, and embeddings**.

### Day 6

**Neural Network From Scratch**
- Train a small network on Moons data.
- Show that loss decreases.
- Perform numerical vs analytic gradient checking.
- Experiment with hidden size, learning rate, and activation.
- Compare against the no-hidden-layer case.

### Day 7

**PyTorch Optimizer Experiments**
- Rebuild the Day 6 network in PyTorch.
- Compare:
  - SGD with learning rates `0.1`, `0.01`, `0.001`
  - Adam with learning rates `0.1`, `0.01`, `0.001`
- Log train/validation loss.
- Plot the runs together.
- Save the trained weights with `state_dict`.

### Day 8

**Mini Project 2 — MNIST, MLP vs CNN**
- Split data as **55k / 5k / 10k**.
- Train an MLP with dropout.
- Train a small CNN.
- Use early stopping.
- Plot training/validation loss.
- Compare accuracy.
- Show a confusion matrix and five incorrect predictions.
- Explain why the CNN behaves differently from the MLP.

### Day 9

**Self-Attention in PyTorch**
- Compute Q, K, and V.
- Implement scaled dot-product attention.
- Verify output shape.
- Verify that attention rows sum to 1.
- Apply a causal mask so future positions receive zero weight.
- Save an attention heatmap.
- Draw a Transformer block and explain Q/K/V in your own words.

### Day 10

**Capstone**
- Build one end-to-end project.
- Keep the code clean and runnable from the README.
- Include a results table, plots, and a six-part explanation.
- Organize the work in one GitHub repository.
- Document mistakes, findings, and next steps.
- Publish a LinkedIn post or blog when appropriate.

## Beginner-to-GenAI Project Ladder

The book provides a progressive ten-stage ladder. Each stage has a concrete “Done when…” condition.

| Stage | Project | Main skills | Done when… |
|---|---|---|---|
| 1 | Data cleaning notebook | NumPy, Pandas, EDA | A messy CSV becomes a clean, documented dataset |
| 2 | Regression / classification | ML, loss, evaluation | A baseline and improved model are compared on validation data |
| 3 | Imbalanced classification | Precision, recall, F1 | You can justify the metric and threshold choice |
| 4 | Semantic similarity | Embeddings, cosine similarity | Similar sentences rank above unrelated ones in your tests |
| 5 | Vector search | Embeddings, FAISS/pgvector concept | Top-5 search over 1,000+ items returns sensible results |
| 6 | PDF RAG assistant | Retrieval, prompting, LLM | Answers cite sources and say “I don’t know” when needed |
| 7 | Tool-using assistant | Tool calling, validation | Two tools work and bad arguments are rejected safely |
| 8 | Stateful agent | Workflow/state, LangGraph concept | A multi-step task completes with a step limit and logs |
| 9 | MCP integration | MCP client/server concepts | Your tool runs as an MCP server used by a client |
| 10 | Production GenAI API | FastAPI, Docker, evaluation | A deployed endpoint has an evaluation set and cost/latency numbers |

The book recommends **not building all ten stages immediately**. Each stage depends on prerequisite concepts.

## Core Mental Models

### Machine learning

```text
Data → Features → Model → Loss → Optimization → Evaluation → Generalization
```

### Deep learning

```text
Input → Linear Layers → Activations → Loss
                      ↑
                 Backpropagation
                      ↑
               Gradient Descent
```

### Transformer / LLM

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token Embeddings + Position
 ↓
Transformer Blocks
 ↓
Vocabulary Logits
 ↓
Next-token Probabilities
 ↓
Choose / Sample Token
 ↓
Append Token
 ↓
Repeat
```

### RAG

```text
Documents
   ↓
Chunking / Embeddings
   ↓
Vector Search
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

### Agentic workflow

```text
Goal
 ↓
LLM decides action
 ↓
Tool call
 ↓
Tool result
 ↓
Updated state
 ↓
Repeat until goal / step limit
```

## Dependency Map

The book's dependency chain is:

```text
Python basics
   ↓
NumPy + Pandas
   ↓
ML concepts + data workflow
   ↓
Loss + gradient descent
   ↓
Evaluation + generalization
   ↓
Neural networks + backprop
   ↓
PyTorch
   ↓
CNNs
   ↓
Embeddings
   ↓
Tokenization
   ↓
Attention
   ↓
Transformers
   ↓
LLM fundamentals
   ↓
RAG
   ↓
Agents
   ↓
Tool calling
   ↓
MCP
   ↓
Evaluation
   ↓
Security
   ↓
FastAPI
   ↓
Docker
   ↓
LLMOps
```

When a topic feels confusing, go back one step in the dependency map rather than memorizing the formula.

## Study Toolkit

Part III is a revision and self-check section containing:

- Formula cheat sheet
- Common-bugs table
- “What comes after” roadmap
- Master beginner checkpoint bank
- Checkpoint answer key
- Revision map for getting unstuck
- Study pattern and interview-ready revision
- Final mental model
- Plain-English glossary
- Transfer learning, augmentation, BatchNorm, and inference supplements
- Classical ML, deep-learning, and GenAI gap supplements
- Line-by-line gradient descent, attention example, and prompt-engineering supplements

## Assignment Companion

The book includes an **Assignment Companion** covering the standard assignment format, acceptance criteria, difficulty labels, dataset setup, GitHub submission layout, project rubric, README template, deliberate debugging, optional extensions, and the project dependency map.

For major projects, the companion uses a 100-point self-check across:

| Criterion | Points |
|---|---:|
| Data preparation | 15 |
| Implementation | 25 |
| Correctness | 20 |
| Evaluation / metrics | 15 |
| Visualization | 10 |
| README | 10 |
| Code quality | 5 |
| **Total** | **100** |

## Suggested GitHub Workflow

Keep the learning work in one repository:

```text
ML-DL-Fundamentals/
├── README.md
├── ML-Deep-Learning-Fundamentals.pdf
├── Assignment-Guide-Hints-Solutions.pdf
└── LICENSE
```

For the actual coding assignments, the book's companion recommends a one-folder-per-day structure such as:

```text
ml-dl-fundamentals/
├── README.md
├── Day-01-NumPy-Pandas/
├── Day-02-Linear-Regression/
├── Day-03-Overfitting/
├── Day-04-Fraud-Detection/
├── Day-05-Telco-Churn/
├── Day-06-Neural-Network/
├── Day-07-PyTorch/
├── Day-08-MNIST/
├── Day-09-Self-Attention/
├── Day-10-Capstone/
├── Project-Ladder/
└── .gitignore
```

For each significant assignment, include a concise README with:

```text
Problem
Dataset
Approach / Architecture
Results (table + plots)
Metrics
Failure cases
What I learned
How to run
Future improvements
```

Do **not** commit large datasets or model weights. Provide download instructions instead.

## Dataset Notes

The book uses or references:

| Dataset / Input | Purpose |
|---|---|
| Titanic | Data cleaning |
| Generated regression data | Linear regression and polynomial fitting |
| Generated Moons data | Neural network from scratch |
| Credit Card Fraud Detection | Imbalanced classification |
| Telco Customer Churn | Customer churn mini project |
| MNIST | MLP vs CNN |

The assignment companion notes the expected filenames and sources for the external datasets. For Telco Churn, the downloaded Kaggle filename may differ from the filename used in the code, so update the path or rename the file consistently.

## Evaluation & Production Mindset

A recurring theme is that a working model is only the beginning.

The material emphasizes:

- Correct train/validation/test separation
- Avoiding data leakage
- Metrics that fit the problem
- Error and failure-case analysis
- Reproducibility
- Clear README documentation
- Safe tool execution
- Evaluation datasets for GenAI systems
- Prompt-injection awareness
- Least-privilege tool access
- Logging and step limits
- Cost and latency awareness before production

For GenAI systems, the book suggests evaluating with real questions and expected answers, including whether a RAG answer is supported by its retrieved context.

## References

The book references official documentation, foundational papers, and dataset sources including:

- NumPy Documentation
- pandas Documentation
- scikit-learn User Guide
- PyTorch Documentation and Tutorials
- LangChain / LangGraph Documentation
- Model Context Protocol Documentation
- Ragas Documentation
- *Attention Is All You Need*
- BERT
- GPT-2
- Adam
- Dropout
- Batch Normalization
- ResNet
- Retrieval-Augmented Generation
- LoRA / QLoRA
- RLHF
- Direct Preference Optimization
- RAGAS
- FAISS
- MNIST
- Kaggle Telco Customer Churn
- Kaggle Credit Card Fraud Detection

See the book's References section for the cited links and full bibliographic details.

## Author

**Medicharla Ravi Kiran**  
AI/ML Engineer & Full-Stack Developer

B.Tech in Computer Science and Engineering, Vaagdevi College of Engineering, Warangal.

- GitHub: https://github.com/Ravikiran9988/
- LinkedIn: https://www.linkedin.com/in/medicharla-ravi-kiran/
- Email: ravikiran@axly.in

## License

Copyright © 2026 Medicharla Ravi Kiran.

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You may share and adapt the material for non-commercial purposes as long as you give appropriate credit to the author and indicate any changes.

Third-party names, libraries, datasets, papers, figures, and trademarks belong to their respective owners and are referenced for educational purposes.

See [LICENSE](LICENSE) for the full license text.

## Important Note

The material is intended for **learning and self-study**. Code examples are kept short so the core ideas remain visible and are provided as-is without warranty. Library APIs can change, so check the official documentation referenced in the material before using examples in a real project.

---

**ML + Deep Learning Fundamentals — Full Beginner → GenAI Edition**  
*First Edition · 2026 · 170 pages*
