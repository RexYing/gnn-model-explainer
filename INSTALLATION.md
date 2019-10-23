# Installation

## From Source

Start by grabbing this source code:

```
git clone https://github.com/RexYing/gnn-model-explainer
```

It is recommended to run this code inside a `virtualenv` with `python3.7`.

```
virtualenv venv -p /usr/local/bin/python3
source venv/bin/activate
```

### Requirements

To install all the requirements, run the following command:

```
python -m pip install -r requirements.txt
```

If you want to install the packages manually, here's what you'll need:


- PyTorch (tested with `1.3`)

```
python -m pip install torch torchvision
```

For alternative methods, check the following [link](https://pytorch.org/)

- OpenCV

```
python -m pip install opencv-python
```

> TODO: It might be worth finding a way to remove the dependency to OpenCV.

- Datascience in Python classics: 

```
python -m pip install matplotlib networkx pandas sklearn seaborn
```

- `TensorboardX`

```
python -m pip install tensorboardX
```

To install [RDKit](https://www.rdkit.org/), follow the instructions provided [on the website](https://www.rdkit.org/docs/Install.html).