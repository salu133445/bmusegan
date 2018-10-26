# Background

## Stochastic and Deterministic Binary Neurons

_Binary neurons_ (BNs) are neurons that output binary-valued predictions. In
this work, we consider two types of BNs:

- _Deterministic Binary Neurons_ (DBNs) act like neurons with hard thresholding
  functions as their activation functions. We define the output of a DBN for a
  real-valued input _x_ as:
  <img src="figs/formula_dbn.png" alt="formula_dbn" style="width:auto; max-height:40px;">
  where __1__<sub>(&middot;)</sub> is the indicator function.

- _Stochastic Binary Neurons_ (SBNs) binarize a real-valued input _x_ according
  to a probability, defined as:
  <img src="figs/formula_sbn.png" alt="formula_sbn" style="width:auto; max-height:40px;">
  where &sigma;(&middot;) is the logistic sigmoid function and _U_[0, 1] denotes
  an uniform distribution.

## Straight-through Estimator

Computing the exact gradients for either DBNs or SBNs, however, is intractable,
for doing so requires the computation of the average loss over all possible
binary samplings of all the BNs, which is exponential in the total number of
BNs.

A few solutions have been proposed to address this issue [1, 2]. In this work,
we resort to the sigmoid-adjusted _straight-through_ (ST) estimator when
training networks with DBNs and SBNs. The ST estimator is first proposed in [3],
which simply treats BNs as identify functions and ignores their gradients. The
sigmoid-adjusted ST estimator is a variant which multiply the gradients in the
backward pass by the derivative of the sigmoid function.

By replacing the non-differentiable functions, which are used in the forward
pass, by differentiable functions (usually called the _estimators_) in the
backward pass, we can then train the whole network with back propagation.

## References

1. Silviu Pitis, "Binary Stochastic Neurons in Tensorflow," 2016.
   Blog post on R2RT blog. &nbsp;
   {% include icon_link.html text="link" icon="fas fa-globe-asia" href="https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html" %}

2. Yoshua Bengio, Nicholas Léonard, and Aaron C. Courville,
   "Estimating or propagating gradients through stochastic neurons for
   conditional computation,"
   _arXiv preprint arXiv:1308.3432_, 2013.

3. Geoffrey Hinton,
   "Neural networks for machine learning - using noise as a regularizer (lecture
   9c)", 2012.
   Coursera, video lecture. &nbsp;
   {% include icon_link.html text="video" icon="fab fa-youtube" href="https://www.coursera.org/lecture/neural-networks/using-noise-as-a-regularizer-7-min-wbw7b" %}