# Model

## Stochastic and Deterministic Binary Neurons

*Binary neurons* (BNs) are neurons that output binary-valued predictions. In
this work, we consider two types of BNs:

- *Deterministic Binary Neurons* (DBNs) act like neurons with hard thresholding
  functions as their activation functions. We define the output of a DBN for a
  real-valued input *x* as:
  <img src="figs/formula_dbn.png" alt="residual_block" style="max-height:30px; display:block;">
  where **1**<sub>(&middot;)</sub> is the indicator function.

- *Stochastic Binary Neurons* (SBNs) binarize a real-valued input *x* according
  to a probability, defined as:
  <img src="figs/formula_sbn.png" alt="residual_block" style="max-height:30px; display:block;">
  where &sigma;(&middot;) is the logistic sigmoid function and *U*[0, 1] denotes
  an uniform distribution.

## Straight-through Estimator

Computing the exact gradients for either DBNs or SBNs, however, is intractable,
for doing so requires the computation of the average loss over all possible
binary samplings of all the BNs, which is exponential in the total number of
BNs.

A few solutions have been proposed to address this issue [1, 2]. In this work,
we resort to the sigmoid-adjusted *straight-through* (ST) estimator when
training networks with DBNs and SBNs. The ST estimator is first proposed in [3],
which simply treats BNs as identify functions and ignores their gradients. The
sigmoid-adjusted ST estimator is a variant which multiply the gradients in the
backward pass by the derivative of the sigmoid function.

By replacing the non-differentiable functions, which are used in the forward
pass, by differentiable functions (usually called the *estimators*) in the
backward pass, we can then train the whole network with back propagation.

## Proposed Model

![bmusegan](figs/bmusegan.png)
<p class="caption" align="center">System diagram of the proposed model</p>

The design is based on MuseGAN [4] as differences summarized below:

- Instead of using multiple generators and use shared/private input vectors to
  handle the multi-track interdependency, we proposed to use shared/private
  generators.

- Instead of using one single discriminator, we proposed to introduce
  shared/private design to the discriminator.

- To help the discriminator extract musically-relevant features, we proposed to
  add to the discriminator an *onset/offset stream* and a *chroma stream*.

### Generator

The generator *G* consists of a shared network <i>G<sub>s</sub></i> followed by
*M* private network <i>G<sub>p</sub><sup>i</sup></i> (*i* = 1 &hellip; *M*), one
for each track. The *shared* network is responsible for producing a high-level
representation of the output musical segments that is shared by all the tracks.
Each *private* network then turns such abstraction into the final piano-roll
output for the corresponding track.

> This design reflects the intuition that different tracks have their own
musical properties (e.g., textures, common-used patterns, playing techniques),
while jointly they follow a common, high-level musical idea.

### Refiner

The refiner *R* is composed of *M* private networks <i>R<sub>i</sub></i>, (*i* =
1 &hellip; *M*), one for each track. The refiner aims to refine the real-valued
outputs of the generator into binary ones rather than learning a new mapping
from *G*(*z*) to the data space. Hence, we draw inspiration from *residual
learning* and propose to construct the refiner with a number of *residual
units*. The output layer of the refiner is made up of either DBNs or SBNs.

<img src="figs/refiner.png" alt="refiner" style="max-height:50px; display:block;">
<p class="caption" align="center">The refiner network.<br>(<i>The tensor size remains the same throughout the network.</i>)</p>

<img src="figs/residual_block.png" alt="residual_block" style="max-height:80px; display:block;">
<p class="caption" align="center">Residual unit used in the refiner network.<br>(<i>The values denote the kernel size and the number of the output channels.</i>)</p>

### Discriminator

Similar to the generator, the discriminator *D* consists of *M* private network
<i>D<sub>p</sub><sup>i</sup></i> (*i* = 1 &hellip; *M*), one for each track,
followed by a shared network <i>D<sub>s</sub></i>. Each private network first
extracts low-level features from the corresponding track of the input
piano-roll. Their outputs are then concatenated and sent to the shared network
to extract higher-level abstraction shared by all the tracks.

In the *onset/offset stream*, the differences between adjacent elements in the
piano-roll along the time axis are first computed, and then the resulting matrix
is summed along the pitch axis, which is finally fed to <i>D<sub>o</sub></i>.

In the *chroma stream*, the piano-roll is viewed as a sequence of one-beat-long
frames. A chroma vector is then computed for each frame and jointly form a
matrix, which is then be fed to <i>D<sub>c</sub></i>.

> Note that all the operations involved in computing the chroma and onset/offset
features are differentiable, and thereby we can still train the whole network by
back propagation.

Finally, the intra-bar features extracted from the three streams are fed to
<i>D<sub>m</sub></i> to extract inter-bar features and to make the final
prediction.

## Reference

1. Binary Stochastic Neurons in Tensorflow, 2016.
   Blog post on R2RT blog.
   [[link](https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html)]

2. Yoshua Bengio, Nicholas Léonard and Aaron C. Courville,
   "Estimating or propagating gradients through stochastic neurons for
   conditional computation," *arXiv preprint arXiv:1308.3432*, 2013.

3. Geoffrey Hinton,
   "Neural networks for machine learning - using noise as a regularizer (lecture
   9c)", 2012.
   Coursera, video lectures.
   [[link](https://www.youtube.com/watch?v=LN0xtUuJsEI)]

4. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang,
   "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic
   Music Generation and Accompaniment,"
   in *AAAI Conference on Artificial Intelligence* (AAAI), 2018.
