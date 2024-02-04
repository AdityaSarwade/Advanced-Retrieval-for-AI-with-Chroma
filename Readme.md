# Advanced Retrieval for AI with Chroma

## Overview

This repository contains code and materials for the "Advanced Retrieval for AI with Chroma" course offered by [deeplearning.ai](https://www.deeplearning.ai/). \
The course focuses on Retrieval Augmented Generative models (RAG) and various topics within it using [ChromaDB](https://www.trychroma.com/) as a vector database. \
The original notebooks use a GPT key to use GPT 3.5 turbo as a Language Model (LLM), but this repository replaces it with an open-source LLM called [StableBeluga](https://huggingface.co/stabilityai/StableBeluga-7B), which can be run locally on a Colab GPU instance. \

PS: Some lesson folders contain gradio implementation for a better user experience

## Contents

- Notebooks: Jupyter notebooks with code implementations for the course.
- Data: Microsoft Annual Repost 2022 as the Sample dataset.
- Utility package: helper_utils.py with utility functions.
- Requirements: requirements.txt to help install necessary packages.

## Usage

- Execute the notebooks to understand and experiment with the concepts covered in the course.
- Refer to the Chroma DB documentation for guidance on utilizing it as a vector database.

## Lessons

1. [Overview of embeddings-based retrieval](./Overview%20of%20embeddings-based%20retrieval/) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XyftsaJNHVapTI75j96sqUBXODV7__bB?usp=sharing)
2. [Pitfalls of retrieval](./Pitfalls%20of%20retrieval/) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kdT-BcqIR5weJf74oaUIaIYPYLfSp0vh?usp=sharing)
3. [Query Expansion](./Query%20Expansion/) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RsJ9kUtr0v4O8qsoLZi6pv9c8RUTFUec?usp=sharing)
4. [Cross Encoder Reranking](./Cross%20Encoder%20Reranking/) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17hoXftJT5ipUm2HLJ5soeRSZUG2R9-kP?usp=sharing)
5. [Embedding Adaptors](./Embedding%20Adaptors/) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zH3t9F8zlFYnyeQoinCCC5S4_2tHxvMY?usp=sharing)

## Acknowledgments

- [deeplearning.ai](https://www.deeplearning.ai/) for providing the "Advanced Retrieval for AI with Chroma" course.
- Contributors to [StableBeluga](https://huggingface.co/stabilityai/StableBeluga-7B) for the open-source LLM.
- The maintainers of [ChromaDB](https://www.trychroma.com/) for the open-source Vector Store.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request. Contributions are welcome!
