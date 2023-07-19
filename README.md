# A Convolutional Neural Network
## by Elias Kramer
### The goal ist to make a library for building a convolutional neural network.

### Implementation
* Layer
  * Fully Connected
    * Forward Propagation
    * Back Propagation
    * Fully GPU supported
  * Convolutional
    * Foward Propagation
    * Backprop is still a TODO
    * Fully supported GPU for every implemented function
  * Pooling
    * Does only work as a standalone
      * integrating it into the nnet is still a TODO
    * Forward Propagation
    * Backprop is still a TODO
    * Fully supported GPU for every implemented function
* Data Space
  * Allocates a huge block of memory for all data you need to train/test a nnet
  * Is implemented as an easy way to store your matrix collection
  * Can be used directly with a Neural Network
  * Pros
    * By allocating one big block of memory for to save a fixed amount of data, it is very space efficient
    * This can be achieved by saving the metadata of a matrix (like format and if it is on the gpu or not) only once
  * Cons
    * A big, contiguous block of memory is often hard to allocate, since you need a lot of free space on your
      RAM or GPU
* Neural Network
  * Can be trained and tested on
    * a Data Space
    * individual matrices
  * Implemented Optimizer
    * Momentum
  * Implemented Initialization
    * Random
    * Xavier
