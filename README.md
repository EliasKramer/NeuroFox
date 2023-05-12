# A Convolutional Neural Network
 ## by Elias Kramer
 ### The goal ist to make a framework for building every convolutional neural network.
 ### Planned Features:
* Fully Connected Layers
* Convolutional Layers
* Pooling Layers
* GPU accelerated learning with CUDA
### Implemetation Details
* The Layers exist separately.
  * Any Layer can set an input an an output format.
  * Every matrix can be propagated forwards and backwards as long as its format is the same as our set input/output format
  * They can propagate any matrix (as long as the format is correct) forward and backward
  * These Layers must be Thread safe (wip)
* Networks have layers. These layers can be shared between networks
  * This saves space if multiple instances of the network are running
  * Networks can propagate a given matrix through their layers and return the output matrix
    * can be done with multiple matrices
