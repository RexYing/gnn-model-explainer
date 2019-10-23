# gnn-explainer

This repository contains the source code for the paper `GNNExplainer: Generating Explanations for Graph Neural Networks` by [Rex Ying](https://cs.stanford.edu/people/rexy/), [Dylan Bourgeois](https://dtsbourg.me/), [Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), [Marinka Zitnik](http://helikoid.si/cms/) & [Jure Leskovec](https://cs.stanford.edu/people/jure/), presented at [NeurIPS 2019](nips.cc).

[[Arxiv]](https://arxiv.org/abs/1903.03894) [[BibTex]](https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1903-03894) [[Google Scholar]](https://scholar.google.com/scholar?q=GNNExplainer%3A%20Generating%20Explanations%20for%20Graph%20Neural%20Networks%20Rex%20arXiv%202019)

```
@misc{ying2019gnnexplainer,
    title={GNNExplainer: Generating Explanations for Graph Neural Networks},
    author={Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
    year={2019},
    eprint={1903.03894},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Using the explainer

### Installation

See [INSTALLATION.md](#)

### Replicating the paper's results

#### Training a GCN model 

This is the model that will be explained. We do provide [pre-trained models](#TODO) for all of the experiments
that are shown in the paper. To re-train these models, run the following:

```
python train.py --dataset=EXPERIMENT_NAME
```

where `EXPERIMENT_NAME` is the experiment you want to replicate. 

For a complete list of options in training the GCN models:

```
python train.py --help
```

> TODO: Explain outputs

#### Explaining a GCN model

To run the explainer, run the following:

```
python explainer_main.py --dataset=EXPERIMENT_NAME
```

where `EXPERIMENT_NAME` is the experiment you want to replicate. 



For a complete list of options provided by the explainer:

```
python train.py --help
```

> TODO: Explain outputs + visualizations + baselines

#### Included experiments

| Name     | `EXPERIMENT_NAME` | Description  |
|----------|:-------------------:|--------------|
| Synthetic #1 | `syn1`  | The first synthetic experiment -- a random BA graph with House attachments.  |
| TODO | TODO  | TODO | 

### Using the explainer on other models


## Changelog

See [CHANGELOG.md](#)