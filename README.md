# Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language models

> [Tian Yu](https://tianyu0313.github.io/), [Shaolei Zhang](https://zhangshaolei1998.github.io/), [Yang Feng](https://people.ucas.edu.cn/~yangfeng?language=en)*

Source code for paper "[Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language models]()".

If you find this project useful, feel free to ⭐️ it and give it a [citation](#citation)!


## Overview

**Auto-RAG** is an autonomous iterative retrieval model centered on the LLM's powerful decision-making capabilities. Auto-RAG models the interaction between the LLM and the retriever through multi-turn dialogue, employs iterative reasoning to determine when to retrieve information and what to retrieve, ceasing the iteration when sufficient external knowledge is available, and subsequently providing the answer to the user.

- **GUI interaction**: We provide a deployable user interaction interface. After inputting a question, Auto-RAG autonomously engages in interaction with the retriever without any human intervention. Users have the option to decide whether to display the details of the interaction between Auto-RAG and the retriever.

<div  align="center">   
  <img src="./assets/auto-rag.gif" alt="img" width="90%" />
</div>


- To interact with Auto-RAG in your browser, follow the guide for [GUI interaction](#gui-interaction).


## Models Download

We provide trained Auto-RAG models using the synthetic data.

## Indexes and Corpus Download

To deploy Auto-RAG, retrieval corpus is required. You could either download our processed [Wikipedia Dump for December 2018] corpus and index, or following [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/building-index.md) to build your own index. 

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
We provide the Wikipedia dump from December 2018 in [Google Drive]().

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

```

If you have any questions, feel free to contact `yutian23s@ict.ac.cn`.
