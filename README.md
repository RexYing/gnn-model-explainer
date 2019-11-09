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

#### Visualizing the explanations

##### Tensorboard

The result of the optimization can be visualized through Tensorboard.

```
tensorboard --logdir log
```

You should then have access to visualizations served from `localhost`.

#### Jupyter Notebook

We provide an example visualization through Jupyter Notebooks in the `notebook` folder. To try it:

```
jupyter notebook
```

The default visualizations are provided in `notebook/GNN-Explainer-Viz.ipynb`.

> Note: For an interactive version, you must enable ipywidgets
>
> ```
> jupyter nbextension enable --py widgetsnbextension
> ```

You can now play around with the mask threshold in the `GNN-Explainer-Viz-interactive.ipynb`.
> TODO: Explain outputs + visualizations + baselines

#### D3,js

We provide export functionality so the generated masks can be visualized in other data visualization 
frameworks, for example [d3.js](http://observablehq.com). We provide [an example visualization in Observable](https://observablehq.com/d/00c5dc74f359e7a1).

#### Included experiments

| Name     | `EXPERIMENT_NAME` | Description  |
|----------|:-------------------:|--------------|
| Synthetic #1 | `syn1`  | Random BA graph with House attachments.  |
| Synthetic #2 | `syn2`  | Random BA graph with community features. | 
| Synthetic #3 | `syn3`  | Random BA graph with grid attachments.  |
| Synthetic #4 | `syn4`  | Random Tree with cycle attachments. |
| Synthetic #5 | `syn5`  | Random Tree with grid attachments. | 
| Enron        | `enron` | Enron email dataset [source](https://www.cs.cmu.edu/~enron/). |
| PPI          | `ppi_essential` | Protein-Protein interaction dataset. |
| | | |
| Reddit*      | `REDDIT-BINARY`  | Reddit-Binary Graphs ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Mutagenicity*      | `Mutagenicity`  | Predicting the mutagenicity of molecules ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Tox 21*      | `Tox21_AHR`  | Predicting a compound's toxicity ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |

> Datasets with a * are passed with the `--bmname` parameter rather than `--dataset` as they require being downloaded manually.

> TODO: Provide all data for experiments packaged so we don't have to split the two.


### Using the explainer on other models
A graph attention model is provided. This repo is still being actively developed to support other
GNN models in the future.

## Changelog

See [CHANGELOG.md](#)
