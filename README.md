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

[1](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
[2](https://www.reddit.com/r/LocalLLaMA/comments/16cdsv6/which_sentence_transformer_is_the_best_one_for/)
[3](https://milvus.io/ai-quick-reference/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2)
[4](https://zilliz.com/ai-faq/how-do-the-various-sbert-models-compare-to-each-other)
[5](https://cholakovit.com/ai/embeddings/sentence-transformers)

