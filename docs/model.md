# Model

<img src="figs/bmusegan.png" alt="system" style="max-width:400px;">

The design is based on [MuseGAN](https://salu133445.github.io/musegan/) [1] as
differences summarized below:

- Instead of using multiple generators and shared/private input vectors to
  handle the multi-track interdependency, we use shared/private generators with
  one single input vector.
- Instead of using one single discriminator, we introduce shared/private design
  to the discriminator.
- To help the discriminator extract musically-relevant features, we add to the
  discriminator an _onset/offset stream_ and a _chroma stream_.

## Generator

The generator _G_ consists of a shared network <i>G<sub>s</sub></i> followed by
_M_ private network <i>G<sub>p</sub><sup>i</sup></i> (_i_ = 1 &hellip; _M_), one
for each track. The _shared_ network is responsible for producing a high-level
representation of the output musical segments that is shared by all the tracks.
Each _private_ network then turns such abstraction into the final piano-roll
output for the corresponding track.

> This design reflects the intuition that different tracks have their own
musical properties (e.g., textures, common-used patterns, playing techniques),
while jointly they follow a common, high-level musical idea.

<img src="figs/generator.png" alt="generator" style="max-width:400px;">

## Refiner

The refiner _R_ is composed of _M_ private networks <i>R<sub>i</sub></i>, (_i_ =
1 &hellip; _M_), one for each track. The refiner aims to refine the real-valued
outputs of the generator into binary ones rather than learning a new mapping
from _G_(_z_) to the data space. Hence, we draw inspiration from _residual
learning_ and propose to construct the refiner with a number of _residual
units_. The output layer of the refiner is made up of either DBNs or SBNs.

<img src="figs/refiner.png" alt="refiner" style="max-width:500px;">
<p class="caption">The refiner network.<br>(<i>The tensor size remains the same throughout the network.</i>)</p>

<img src="figs/residual_block.png" alt="residual_block" style="max-width:300px;">
<p class="caption">Residual unit used in the refiner network.<br>(<i>The values denote the kernel size and the number of the output channels.</i>)</p>

## Discriminator

Similar to the generator, the discriminator _D_ consists of _M_ private network
<i>D<sub>p</sub><sup>i</sup></i> (_i_ = 1 &hellip; _M_), one for each track,
followed by a shared network <i>D<sub>s</sub></i>. Each private network first
extracts low-level features from the corresponding track of the input
piano-roll. Their outputs are then concatenated and sent to the shared network
to extract higher-level abstraction shared by all the tracks.

In the _onset/offset stream_, the differences between adjacent elements in the
piano-roll along the time axis are first computed, and then the resulting matrix
is summed along the pitch axis, which is finally fed to <i>D<sub>o</sub></i>.

In the _chroma stream_, the piano-roll is viewed as a sequence of one-beat-long
frames. A chroma vector is then computed for each frame and jointly form a
matrix, which is then be fed to <i>D<sub>c</sub></i>.

> Note that all the operations involved in computing the chroma and onset/offset
features are differentiable, and thereby we can still train the whole network by
back propagation.

Finally, the intra-bar features extracted from the three streams are fed to
<i>D<sub>m</sub></i> to extract inter-bar features and to make the final
prediction.

<img src="figs/discriminator.png" alt="discriminator" style="max-width:500px;">

## Reference

1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
   "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic
   Music Generation and Accompaniment,"
   in _AAAI Conference on Artificial Intelligence_ (AAAI), 2018.
