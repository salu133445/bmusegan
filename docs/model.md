# Model

![bmusegan](figs/bmusegan.png)

The design is based on [MuseGAN](https://salu133445.github.io/musegan/) [1] as differences summarized below:

- Instead of using multiple generators and use shared/private input vectors to
  handle the multi-track interdependency, we proposed to use shared/private
  generators.
- Instead of using one single discriminator, we proposed to introduce
  shared/private design to the discriminator.
- To help the discriminator extract musically-relevant features, we proposed to
  add to the discriminator an *onset/offset stream* and a *chroma stream*.

## Generator

The generator *G* consists of a shared network <i>G<sub>s</sub></i> followed by
*M* private network <i>G<sub>p</sub><sup>i</sup></i> (*i* = 1 &hellip; *M*), one
for each track. The *shared* network is responsible for producing a high-level
representation of the output musical segments that is shared by all the tracks.
Each *private* network then turns such abstraction into the final piano-roll
output for the corresponding track.

> This design reflects the intuition that different tracks have their own
musical properties (e.g., textures, common-used patterns, playing techniques),
while jointly they follow a common, high-level musical idea.

<img src="figs/generator.png" alt="generator" style="max-height:200px; display:block;">

## Refiner

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

## Discriminator

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

<img src="figs/discriminator.png" alt="discriminator" style="max-height:300px; display:block;">

## Reference

1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang,
   "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic
   Music Generation and Accompaniment,"
   in *AAAI Conference on Artificial Intelligence* (AAAI), 2018.
