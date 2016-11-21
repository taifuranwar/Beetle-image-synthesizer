beetle image synthesizer
----------------------

This library synthesize image using vgg16 feature model set and dCNN to project the image correctly for main NN of Beetle application.



This is image synthesizer library of beetle application. If you want to learn more about beetle application, click here.

In short, it is an windows phone application that detect crop diseases. Put the infected leaf in a piece of white paper and use the application. Beetle will take an image of the leaf, send it to the server. The server then do the image processing and recognition with the existing database and detect the disease. 

One of the core problem we had while making Beetle is it was very hard to detect infected area and reconstruct it for the Neural Network so that while training/detection, it would not contain noises and uninfected portion of the leaf. Also while training and detecting, it is very crucial to make the orientation of the infected portion of the image right. The user doesn't follow the way to take pictures and the outcome is the detection picture comes in many varieties.

Beetle image synthesizer library solves this problems by extracting features of the image. It uses gaussian pyramid and feature vector of trained data, then it uses ANN ( Approximate nearest neighbour search ). The synthesis then proceeds from coarsest resolution to finest, computing a multiscale representation of output image. 

We have used Markov random field (MRF) models and discriminatively trained deep convolutional neural networks (dCNNs) for synthesizing the images. This part of the project, along with patch matching and blending is inspired by arXiv's project Please refer to [this github project]( http://www.gitxiv.com/posts/DtC4Zwz3kqCDBHFD7/combining-markov-random-fields-and-convolutional-neural) for more information.

![banner](https://raw.githubusercontent.com/taifuranwar/Beetle-image-synthesizer/master/images/banner.jpg)

We have also used vgg16 model for feature map.Original file can be downloaded from  [this github project]( https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) 



Installation
------------
This requires either  [TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html) or [Theano](http://deeplearning.net/software/theano/install.html). If you don't have a GPU you'll want to use TensorFlow. GPU users may find to Theano to be faster at the expense of longer startup times. Here's the [Theano GPU guide]( http://deeplearning.net/software/theano/tutorial/using_gpu.html).

Here's how to [configure the backend with Keras](http://keras.io/backend/) and set your default device (e.g. cpu, gpu0).

Also, make sure that you have these python package installed: 

        'Cython>=0.23.4',
        'h5py>=2.5.0',
        'numpy>=1.10.4',
        'Pillow>=3.1.1',
        'PyYAML>=3.11',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'six>=1.10.0',

If you have trouble with the above method, follow these directions to [Install latest keras and theano or TensorFlow](http://keras.io/#installation)

The script `imageSynthesizer.py`  is the main script.

**Before running this script**, download the [weights for the VGG16 model](
https://github.com/awentzonline/image-analogies/releases/download/v0.0.5/vgg16_weights.h5). This file contains only the convolutional layers of VGG16 which is 10% of the full size. [Original source of full weights](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).
The script assumes the weights are in the current working directory. If you place
them somewhere else make sure to pass the `--vgg-weights=<location-of-the-weights.h5>` parameter 

Example script usage:
`imageSynthesizer.py sourceImageMask sourceImage targetImage targetImageMask`

e.g.:

`imageSynthesizer.py /home/saif/venv_1/images/sourceImages/1_image.jpg /home/saif/venv_1/images/sourceImages/1_mask.jpg /home/saif/venv_1/images/targetImageMasks/1_mask.jpg /home/saif/venv_1/out/arch --vgg-weights /home/saif/venv/vgg16_weights.h5`