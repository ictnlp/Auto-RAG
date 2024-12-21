# Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language models

> [Tian Yu](https://tianyu0313.github.io/), [Shaolei Zhang](https://zhangshaolei1998.github.io/), [Yang Feng](https://people.ucas.edu.cn/~yangfeng?language=en)*

[![arXiv](https://img.shields.io/badge/arXiv-2411.19443-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19443)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/ictnlp/Auto-RAG)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/ICTNLP/Auto-RAG)

Source code for paper "[Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language models](https://arxiv.org/abs/2411.19443)".

If you find this project useful, feel free to ⭐️ it and give it a [citation](#citation)!


## Overview

**Auto-RAG** is an autonomous iterative retrieval model centered on the LLM's powerful decision-making capabilities. Auto-RAG models the interaction between the LLM and the retriever through multi-turn dialogue, employs iterative reasoning to determine when and what to retrieve, ceasing the iteration when sufficient external knowledge is available, and subsequently providing the answer to the user.

- **GUI interaction**: We provide a deployable user interaction interface. After inputting a question, Auto-RAG autonomously engages in interaction with the retriever without any human intervention. Users have the option to decide whether to display the details of the interaction between Auto-RAG and the retriever.

<div  align="center">   
  <img src="./assets/auto-rag.gif" alt="img" width="90%" />
</div>


- To interact with Auto-RAG in your browser, follow the guide for [GUI interaction](#gui-interaction).


## Models Download

We provide trained Auto-RAG models using the synthetic data. Please refer to https://huggingface.co/ICTNLP/Auto-RAG-Llama-3-8B-Instruct.

## Indexes and Corpus Download

To deploy Auto-RAG, retrieval corpus is required. You can download from official [website](https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2) or our processed corpus (which will be uploaded soon), and following [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/building-index.md) to build your own index. 

## Installation

- Clone Auto-RAG's repo.

```bash
git clone https://github.com/ictnlp/Auto-RAG.git
cd Auto-RAG
export ROOT=pwd
```

- Environment requirements: Python 3.12, Gradio 5.1.0

```bash
conda env create -f environment.yml
```

- Download indexes and corpus

We used the following dump version: https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2. Please follow the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/process-wiki.md) to process and index it.


## Model deployment

We use vLLM to deploy the model for inference. You can update the parameters in vllm.sh to adjust the GPU and model path configuration, then execute:

```bash
bash vllm.sh
```


## GUI Interaction

To interact with Auto-RAG in your browser, you should firstly [download](#models-download) the trained Auto-RAG Models and prepare for [retrieval corpus](#indexes-and-corpus-download).

```bash
cd $ROOT/webui
CUDA_VISIBLE_DEVICES=0,1,2,3 python webui.py\
    --main_model {model_name}\
    --main_model_url {main_model_url}\
    --dense_corpus_path {dense_corpus_path}\
    --dense_index_path {dense_index_path}

```

> [!Tip]
> The interaction process between Auto-RAG and the retriever can be optionally displayed by adjusting a toggle.

## Evaluation
> [!Note]
> Experimental results show that Auto-RAG outperforms all baselines across six benchmarks.

<div  align="center">   
  <img src="./assets/results.png" alt="img" width="100%" />
</div>
<p align="center">

</p>


## Licence


## Citation

If this repository is useful for you, please cite as:

```
@article{yu2024autorag,
      title={Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models}, 
      author={Tian Yu and Shaolei Zhang and Yang Feng},
      year={2024},
      eprint={2411.19443},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.19443}, 
}
```

If you have any questions, feel free to contact `yutian23s@ict.ac.cn`.
