# gnn-explainer

This repository contains the source code for the paper `GNNExplainer: Generating Explanations for Graph Neural Networks` by Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec, presented at NeurIPS 2019.

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

| Name | `EXPERIMENT_NAME` | Description  |
|------|-------------------|--------------|
| Synthetic #1 | `syn1`  | The first synthetic experiment -- a random BA graph with House attachments.  |
| TODO | TODO  | TODO | 

### Using the explainer on other models


## Changelog

See [CHANGELOG.md](#)