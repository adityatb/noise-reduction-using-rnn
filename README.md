# Noise Reduction using RNNs with Tensorflow
Implements python programs to train and test a Recurrent Neural Network with Tensorflow. This program is adapted from the methodology applied for Singing Voice separation, and can easily be modified to train a source separation example using the MIR-1k dataset.

References: Huang, Po-Sen, Minje Kim, Mark Hasegawa-Johnson, and Paris Smaragdis. "Singing-Voice Separation from Monaural Recordings using Deep Recurrent Neural Networks." In ISMIR, pp. 477-482. 2014

## Instructions
Before running the programs, some pre-requisites are required. This code is developed for Python 3, with numpy, and scipy (v0.19) libraries installed. In addition, Tensorflow v1.2 is required. The code is setup to be executable directly on FloydHub servers using the commands in the comments at the top of the script.

This project additionally relies on the MIR-1k dataset, which isn't packed into this git repo due to its large size. It can be downloaded here freely: http://mirlab.org/dataSet/public/MIR-1K_for_MIREX.rar

If running on FloydHub, the complete MIR-1K dataset is already publicly available at:
https://www.floydhub.com/adityatb/datasets/mymir/2:mymir

A shorter version of the dataset is also available for debugging, before deploying completely:
https://www.floydhub.com/adityatb/datasets/mymir/1:mymir

The scripts are Tensorboard active, so you can track accuracy and loss in realtime, to evaluate the training. The data written to the logs folder is read by Tensorboard.

If running on your local machine, the MIR-1k dataset will need to be downloaded and setup one level up:
Dataset: "../input/mir1k/MIR-1k/"
Noises: "../input/mir1k/MIR-1k/Noises"

Lastly: TrainNet.py runs the training on the dataset and logs metrics to TensorBoard.

TrainNetBSS runs trains a singing voice separation experiment.
