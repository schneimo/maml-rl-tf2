# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2
Implementation of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in TensorFlow 2. 

This repo was heavily inspired by the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) and a very clear implementation from students at MILA [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl). I totally recommend to check out both implementations too.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with MAML. This script was tested with Python 3.6. Note that some environments may also work with Python 2.7 (all experiments besides MuJoCo-based environments).
```
python main.py --env-name HalfCheetahDir-v1 --num-workers 8 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 1000 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-halfcheetah-dir --device cuda
```

## References
This project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in TensorFlow 2. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite this paper
```
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```