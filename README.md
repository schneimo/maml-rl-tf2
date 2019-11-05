# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) in TensorFlow 2 (WIP)
*Currently* working on the implementation of *Model-Agnostic Meta-Learning (MAML)* applied on Reinforcement Learning problems in TensorFlow 2. 

This repo is heavily inspired by the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) and the very clear implementations of Tristan Deleu from MILA [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl) (PyTorch) and Jonas Rothfuss [jonasrothfuss/ProMP](https://github.com/jonasrothfuss/ProMP) (Tensorflow 1). 

I totally recommend to check out all three implementations too.

## Work in Progress
**This repo is currently *under construction* and not useable at the moment.** 

*Current State*: Implementing the basic MAML algorithm with TRPO as the optimization method.

*Future*: Later this repo should include the basic MAML algorithm and also other variations like Reptile, ProMP, etc.

## Usage
You can use the [`main.py`](main.py) script in order to train the algorithm with MAML.
```
python main.py --env-name 2DNavigation-v0 --num-workers 20 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 20 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 500 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --device cuda
```
This script was tested with Python 3.6.

## References
This project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in TensorFlow 2. The experiments are based on the paper
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
