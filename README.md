# Python Code Analysis & Embedding Demos
This repository demonstrates three essential steps for analyzing Python code snippets using modern libraries:

Static Structure Analysis with AST and Tokenization

## Tokenization Using DistilRoBERTa

Generating Semantic Embeddings Using Pretrained Models

## 1. Static Code Structure Analysis
This code inspects Python code snippets to extract useful information—without executing them!

Finds: Function names, class names, imported modules.

Counts: Specific coding patterns (like comprehensions, attribute access, with-statements, etc.).

Tokenizes: Breaks code into tokens (words, symbols) for further analysis.

<details> <summary>Show sample code</summary>
python
import ast
import tokenize
import io

class CodeAnalyzer(ast.NodeVisitor):
    # ... (Class definition code)
    pass

def tokenize_code(code):
    # ... (Tokenization code)
    pass

snippets = [
    # ... (Your code snippets)
]

for code in snippets:
    # ... (Analysis code)
    pass
</details>
Purpose
Analyze code structure, learn what constructs are used, and prepare for deeper analysis.

## 2. Tokenizing Code Snippets (DistilRoBERTa)
This code converts Python code snippets from text into token IDs using the DistilRoBERTa tokenizer.

Tokenization: Splits code into model-ready pieces.

Padding/Truncation: Ensures all snippets are the same length.

Attention Masks: Marks which parts are real code (not padding).

<details> <summary>Show sample code</summary>
python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

encoded_inputs = tokenizer(
    snippets,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
# ... (Printing results)
</details>
Purpose
Makes code understandable for AI models by turning text into numbers.

## 3. Creating Semantic Embeddings of Code
This code uses pretrained models to generate vector representations (embeddings) of code snippets. Supported models:

MiniLM

DistilRoBERTa

MPNet

Embeddings: Turn code into fixed-size numeric vectors.

Comparison: These vectors can be used to compare code snippets based on meaning and structure.

<details> <summary>Show sample code</summary>
python
from sentence_transformers import SentenceTransformer

model_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_distilroberta = SentenceTransformer('sentence-transformers/msmarco-distilroberta-base-v2')
model_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

embeddings_minilm = model_minilm.encode(snippets, normalize_embeddings=True)
embeddings_distilroberta = model_distilroberta.encode(snippets, normalize_embeddings=True)
embeddings_mpnet = model_mpnet.encode(snippets, normalize_embeddings=True)
</details>
Purpose
Capture the semantic meaning of code for search, clustering, or advanced code analysis.

### How To Use
1.Clone this repository.

2.Install requirements:

pip install transformers sentence-transformers

3.Replace the example snippets with your Python code snippets.

4.Run the Jupyter notebook or Python scripts.

## Why These Steps Matter
Static analysis is the foundation for code understanding, audits, and safety checks.

Tokenization bridges the gap between raw code and machine learning models.

Embeddings let you build smarter tools that understand and find code by meaning, not just by text.
![generated-image](https://github.com/user-attachments/assets/dfdc904e-2d47-4924-b15a-6d8cc2d37f5c)
Here is a clear comparison of the three embedding models used:

| Model                        | Main Focus           | Size & Speed         | Embedding Quality        | Best Use Cases                             |
|------------------------------|----------------------|----------------------|-------------------------|--------------------------------------------|
| **MiniLM (all-MiniLM-L6-v2)**| Fast & Lightweight   | Small (384 dims), very fast, low memory use | Good (but not top-tier)   | Real-time apps, mobile, high-volume search |
| **DistilRoBERTa (msmarco-distilroberta-base-v2)** | Search Optimized      | Medium size (768 dims), moderate speed     | High (training for relevance/ranking)      | Search, retrieval, recommender systems     |
| **MPNet (all-mpnet-base-v2)**| Highest Accuracy     | Larger (768 dims), slower, higher memory use| Best (latest architecture)                 | Deep semantic search, clustering, analysis |

***

### Key Details

- **MiniLM** is the smallest and fastest, making it ideal where speed matters most (like mobile apps and real-time systems). It sacrifices a bit of accuracy for efficiency.
- **DistilRoBERTa** balances speed and quality, and is fine-tuned for searching and relevance-ranking tasks. It’s a good middle ground when you want strong accuracy but don’t need maximum precision.
- **MPNet** uses the newest transformer architecture for the best accuracy in catching the precise meaning of text or code. It requires more memory and runs slower, but is chosen when top semantic performance is most important.

***

### When to Use Each Model

- Choose **MiniLM** for quick, large-scale tasks (chatbots, autocomplete).
- Choose **DistilRoBERTa** when building a search engine or recommendation tool that relies on finding relevant code or text.
- Choose **MPNet** for deep analysis, clustering, or any application where understanding deeper meaning and achieving the highest accuracy is the main goal.



# MILESTONE2
# Code Generation & Complexity Analysis using Hugging Face Transformers

This project demonstrates how to generate Python code automatically from natural language prompts using state-of-the-art Hugging Face transformer models, and then evaluate the complexity and maintainability of the generated code using Radon.

## 🚀 Features

Interactive Jupyter interface with ipywidgets

Automatic code generation using Hugging Face language models

Code analysis metrics:

Cyclomatic Complexity

Maintainability Index

Lines of Code (LOC)

Automatic results logging in a Pandas DataFrame

## 🧩 Tech Stack
Component	Library
Language Model	🤗 transformers
Deep Learning Backend	torch
Code Metrics	radon
UI Components	ipywidgets, IPython.display
Data Handling	pandas

## 📦 Installation

Clone the repository and install the dependencies:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

pip install torch transformers ipywidgets radon pandas

## 🔑 Setup

Before running the notebook, create a Hugging Face token:

Visit https://huggingface.co/settings/tokens

Copy your personal access token.

Add it to your notebook:

HFTOKEN = "<your_huggingface_token_here>"

## 🧮 Usage

Open the notebook milestone2.ipynb in Jupyter Notebook or Google Colab.

Run all the cells.

At the bottom, call the model interface function with your desired model:

model_cell("CodeLlama 7B", "codellama/CodeLlama-7b-hf")


Enter a prompt (e.g., “Write a Python function to compute factorial of a number”) and click Generate.

The model will:

Generate code

Display it in the output

Show metrics:

Cyclomatic Complexity

Maintainability Index

Lines of Code

All results are stored in the global metrics_df DataFrame.

.

## 🧠 Example Output
Prompt:
# Write a Python function to compute factorial of a number.

Generated Code:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

Metrics:
  Cyclomatic Complexity: 2
  Maintainability Index: 84.9
  Lines of Code (LOC): 5

## 📊 DataFrame Example
Model	Prompt	Cyclomatic Complexity	Maintainability Index	LOC
CodeLlama 7B	Write factorial function	2	84.9	5
## 💡 Tips

You can use any compatible Hugging Face model, e.g.:

model_cell("StarCoder", "bigcode/starcoder")


Adjust max_new_tokens and temperature for more or less creative outputs.


