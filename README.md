This project uses a convolutional neural network (CNN) to classify MRI scans of brain tumors. The network is built in PyTorch, trained to identify whether an MRI scan contains a brain tumor. The dataset used
in this project is provided: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data. This dataset includes training and testing data from MRI scans with associated labels. 

This code defines and trains a custom CNN model, Stats453Classifier, to classify MRI images into four categories based on the presence of brain tumors. Here is a breakdown of the model’s main components:

Convolutional Layers: Used for feature extraction with ReLU activations and max-pooling layers.
Fully Connected Layers: Used for classification, ending with a final layer matching the number of output classes.
Dropout Layer: Helps reduce overfitting by randomly setting a fraction of input units to zero at each update.

