# neural-quines
A collection of "neural network quines," networks that ouptut their own weights (with varying degrees of success).

Inspired by https://arxiv.org/abs/1803.05859, and discussed in much greater detail here: https://evanfletcher42.com/2022/10/31/neural-quines/


![A plot comparing neural network output to the network's own weights.](https://github.com/evanfletcher42/neural-quines/blob/main/figures/img_approx_bn_1.png)

## Requirements
Pytorch, numpy, and matplotlib.  

## Quines

Each file is self-contained, with all the code to define, train, and visualize the performance of each net.  

 - **quine_image_approx_coords.py**: A simple network with one trainable fully-connected layer, where weights can be queried by normalized (row, column) index.  Rapidly converges to a trivial "zero quine," which - while perfect - is also boring.  
 - **quine_image_approx_coords_normalized.py**: Like the previous network, but uses a parameter-free normalization layer to forcibly avoid the zero quine.  Converges to a non-trivial quine with small error.  
 - **quine_onehot_matrix.py**: Similar to the previous quine, except that inputs are encoded as a one-hot matrix.  Converges to a non-trivial quine with small-ish error.
 - **quine_recurrent_weights.py**: A RNN that outputs all of the network's weights in one long sequence.  Converges to an almost-trivial result (most weights are a constant).  
 - **quine_recurrent_source.py**: A character-level RNN that outputs the source code that defines & trains the RNN.  Technically cheating, but fun regardless.  
