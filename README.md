# Variational Dropout Sparsifies DNN (Pytorch implementation)
Pytorch implementation of Variational Dropout Sparsifies Deep Neural Networks ([arxiv:1701.05369](https://arxiv.org/abs/1701.05369)).

## Description
The discovered approach helps to train both convolutional and dense deep sparsified models without significant loss of quality. Additive Noise Reparameterization
and the Local Reparameterization Trick discovered in the paper helps to eliminate weights prior's restrictions ($\alpha \leq 1$) and achieve Automatic Relevance Determination (ARD) effect on (typically most) network's parameters. According to the original paper, authors reduced the number of parameters up to 280 times on LeNet architectures and up to 68 times on VGG-like networks with a negligible decrease of accuracy. Experiments with Boston dataset in this repository proves that: 99% of simple dense model were dropped using paper's ARD-prior without any significant loss of MSE. Moreover, this technique helps to significantly reduce overfitting and helps to not worry about model's complexity - all redundant parameters will be dropped automatically. Moreover, you can achieve any degree of regularization variating regularization factor tradeoff (see ***reg_factor*** variable in [boston_ard.py](examples/boston_ard.py) script)

## Requirements
* **PyTorch** >= 0.4.0
* **SkLearn** >= 0.19.1
* **Pandas** >= 0.23.3

## Example
```
cd examples
python boston_ard.py
```

## Benchmarks

### Boston dataset

Two scripts were used in the experiment: [boston_baseline.py](examples/boston_baseline.py) and [boston_ard.py](examples/boston_ard.py). Training procedure for each experiment was **100000 epoches, Adam(lr=1e-3)**. Baseline model was dense neural network with single hidden layer with hidden size 150.

|                | Baseline (nn.Linear) | LinearARD, no reg | LinearARD, reg=0.0001 | LinearARD, reg=0.001 | LinearARD, reg=0.1 | LinearARD, reg=1 |
|----------------|----------|-------------|-----------------|----------------|--------------|------------|
| MSE (train)    | 1.751    | 1.626       | <span style="color:green"><b>1.587</b></span>           | 1.962          | 17.167       | 33.682     |
| MSE (test)     | <span style="color:red"><b>22.580</b></span>   | 16.229      | 20.950          | <span style="color:green"><b>8.416</b></span>          | 25.695       | 30.231     |
| Compression, % | <span style="color:red"><b>0</b></span>        | 0.38        | <span style="color:green"><b>52.95</b></span>           | <span style="color:green"><b>64.19</b></span>          | <span style="color:green"><b>97.29</b></span>        | <span style="color:green"><b>99.29</b></span>      |

You can see on the table above that variating regularization factor any degree of compression can be achieved (for example, ~99.29% of connections can be dropped if reg_factor=1 will be used). Moreover, you can see that training with LinearARD layers with some regularization parameters (like reg=0.001 in the table above) not only significantly reduces number of model parameters (>64% of parameters can be dropped after training), but also significantly increases quality on test, reducing overfitting.



## TODO
- [X] LinearARD layer implementation
- [ ] Conv2DARD layer implementation

## Authors

```
@article{molchanov2017variational,
  title={Variational Dropout Sparsifies Deep Neural Networks},
  author={Molchanov, Dmitry and Ashukha, Arsenii and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:1701.05369},
  year={2017}
}
```
[Source code](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn) (Theano/Lasagne implementation)

## Contacts

Artem Ryzhikov, LAMBDA laboratory, Higher School of Economics, Yandex School of Data Analysis

**E-mail:** artemryzhikoff@yandex.ru

**Linkedin:** https://www.linkedin.com/in/artem-ryzhikov-2b6308103/

**Link:** https://www.hse.ru/org/persons/190912317
