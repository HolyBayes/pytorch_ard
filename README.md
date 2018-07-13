# VarDropPytorch
Pytorch implementation of Variational Dropout Sparsifies Deep Neural Networks ([arxiv:1701.05369](https://arxiv.org/abs/1701.05369)). The discovered approach helps to train both convolutional and dense deep sparsified models without significant loss of quality. Additive Noise Reparameterization
and the Local Reparameterization Trick discovered in paper helps to eliminate weights prior's restrictions ($\alpha \leq 1$) and achieve Automatic Relevance Determination (ARD) effect on most network's connections. According to the original paper, authors reduced the number of parameters up to 280 times on LeNet architectures and up to 68 times on VGG-like networks with a negligible decrease of accuracy. Experiments with Boston dataset in this repository proves that: 99% of simple dense model were dropped using paper's ARD-prior without any significant loss of MSE. Moreover, this technique helps to significantly reduce overfitting and helps to not worry about model's complexity - all redundant parameters will be dropped automatically. Moreover, you can achieve any degree of regularization variating regularization factor tradeoff (see *reg_factor* variable in examples/boston.py script)

## Requirements
* **PyTorch** >= 0.4.0
* **SkLearn** >= 0.19.1
* **Pandas** >= 0.23.3

## Example
```
cd examples
python boston.py
```
## TODO
- [X] Linear layer implementation
- [ ] Conv layers implementation

## Authors

```
@article{molchanov2017variational,
  title={Variational Dropout Sparsifies Deep Neural Networks},
  author={Molchanov, Dmitry and Ashukha, Arsenii and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:1701.05369},
  year={2017}
}
```
[Source code](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn) (Original Theano/Lasagne implementation)
