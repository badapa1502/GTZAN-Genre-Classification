# GTZAN-Genre-Classification
The goal is to train and experiment with different CNN/RNN achitecture based classifier on the Mel spectrograms from the GTZAN dataset to predict the corresponding music genres.

## GTZAN Dataset: modified
The data represent the log-transformed Mel spectrograms derived from the GTZAN dataset. The original GTZAN dataset contains 30-seconds audio files of 1,000 songs associated with 10 different genres (100 per genre). We have reduced the original data to 8 genres (800 songs) and transformed it to obtain, for each song, 15 log-transformed Mel spectrograms. Each Mel spectrogram is an image file represented by a tensor of shape (80, 80, 1) which describes time, frequency and intensity of a song segment. The training data represent 80% of the total number of data points. Download the [training and validation data](https://drive.google.com/drive/folders/154dxA9DPaEUW_QbCuW8e-eA5xYd73CQL?usp=sharing).

# P1 - Parallel CNNs and RNNs

## P1.1: Shallow Parallel CNN architecture
- First parallel branch:
  1. one convolutional layer processing the input data with 3 square filters of size 8, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with pooling size 4.
  3. a layer flattening the output of the pooling.

- Second parallel branch:
  1. one convolutional layer processing the input data with 4 square filters of size 4, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with size 2.
  3. a layer flattening the output of the pooling..

- Merging branch:
  1. a layer concatenating the outputs of the two parallel branches.
  2. a dense layer which performs the classification of the music genres using the approppriate activation function.

Training method: Mini-batch stochastic gradient descent algorithm, trained for 50 epochs.
Evaluation: Plots for the loss function and the accuracy versus the number of epochs for the train and validation sets provided.

## P1.2: CNN-RNN based classifier
First, you need to reduce the dimensionality of the dataset by running the following code:

`def reduce_dimension(x, y):`

`  return tf.squeeze(x, axis=-1), y`

`train_dataset_squeeze = train_dataset.map(reduce_dimension)`

and similarly on the validation set. Then, batch the resulting train and validation sets with batch size of 128.

CNN-RNN architeture:
1. a convolutional layer with 8 filters of size 4.
2. a max pooling layer that halves the dimensionality of the output.
3. a convolutional layer with 6 filters of size 3.
4. a max pooling layer that halves the dimensionality of the output.
5. an LSTM layer with 128 units, returning the full sequence of hidden states as output.
6. an LSTM layer with 32 units, returning only the last hidden state as output.
7. a dense layer with 200 neurons and ReLU activation function.
8. a layer dropping out 20% of the neurons during training.
9. a dense layer which outputs the probabilities associated to each genre.

Training method: Mini-batch stochastic gradient descent algorithm, trained for 50 epochs.
Evaluation: Plots for the loss function and the accuracy versus the number of epochs for the train and validation sets provided.

# P2 - Achieving higher accuracy
Different data augmentation techniques, CNNs, RNNs, mix of CNN/RNNs, optimisers/learning rates were adopted.

## Wider, Deeper Parallel CNN: varying filter sizes to learn more detailed frequency patterns with batch normalisation and dropout for better generalisation
The model has three parallel branches with different convolutional filter sizes to capture musical features at multiple levels of granularity. Each branch specialises in detecting patterns at different resolutions, improving feature extraction.
1. Branch 1 (Broad Patterns): Uses large 8x8 filters to capture broad frequency structures. filters to capture broad harmonic structures and large-scale features. Max pooling with 4x4 window ensures strong downsampling, reducing redundant information.
2. Branch 2 (Finer Details): Uses 4x4 filters to extract finer musical textures and rhythmic structures. Smaller max pooling window 2x2 preserves more detailed information.
3. Branch 3 (Small Receptive Fields): Uses 2x2 filters to focus on fine-grained details, e.g. timbre or dynamics in spectrograms.

We also apply batch normalisation to each branch to stabilise learning, and then concatenate the branches. A dense layer with 256 neurons learns complex non-linear combinations of the extracted features, while a dropout (0.3) layer prevents overfitting by randomly deactivating neurons during training.

## ResNet-50
To consider an alternate CNN architecture, I tried to implement a ResNet-50 model on the processed GTZAN dataset, by switching the weights to "None" instead of ImageNet, such that the ResNet model is trained from scratch. This architectural choice was made because while spectrograms visually resemble images, they represent a time-frequency transformation of audio data; different from the natural images that ImageNet weights are trained on. However, this model had very poor training and validation performance on the dataset, with validation accuracy around 25% even at the 50th epoch. This could be because ResNet-50 was originally designed for natural image classification, not to capture temporal / sequential patterns found in audio data. I did try transfer learning using ImageNet weights, but as expected the model still struggled because audio features and patterns differen from the features found in natural images.

## RNN Architectures
I tried LSTM and BiLSTM as a standalone RNN architectures (not combined with CNN), but their performance plateaued at 50-60% accuracy on the GTZAN dataset. I believe the main reason is because the GTZAN dataset originally consists of 30-second audio clips for each song; but since we use preprocessed Mel spectrograms in which each song has been divided into 15 segments, the final duration of each audio segment in the dataset is 2 seconds. Given this short duration of the spectrogram, it may not give LSTM enough temporal context to extract meaningful temporal dependencies from the data; it is typically better learned over longer sequences. It could also be that spatial features are more relevant for genre classification than temporal classification.

# CNN + RNN Architectures
We tried 3 hybrid models that combine CNNs for spatial feature extraction with RNNs (like LSTM or GRU) for any remaining temporal dependencies might offer better performance by capturing both the spatial and temporal aspects of the data. Note that Conv1D is traditionally used for sequential data in CNN + RNN architectures, to capture patterns along the time dimension. However, by replacing Conv1D with Conv2D, it gave me better results because our input data, spectrograms, are not just sequential data but also spatial data (frequencies). Hence, Conv2D allowed for better capturing of both spatial and time relationships, and we proceed with that from now on for our model architectures below, and reshape the output from the CNN accordingly from 3D output to 2D input for the RNN models shown below: 

## CNN + BiLSTM
Convolutional Layers (Feature Extraction): The 3 convolutional layers followed by max pooling and batch normalisation help capture varying levels of non-redundant, spatial frequency patterns without overfitting.
Reshaping for LSTM Input: The output from the CNN is reshaped to (10, 640), treating the time axis (10) as sequence steps for LSTMs.
Bidirectional LSTM Layers (Temporal Feature Learning): captures temporal dependencies in both forward and backward direction - by considering both the past and future musical context.
Fully Connected Layers: to learn complex feature interactions, followed by dropout layer to reduce dimensionality and prevent overfitting.

## Data Augmented CNN + GRU
### Model architecture:
In terms of model architecture, we use the same layout as CNN + BiLSTM model, except we replace the 2 BiLSTM layers with 2 GRU layers. We consider GRU because GRUs are similar to LSTMs but with fewer parameters because they use fewer gates. This makes GRUs faster to train and more efficient, especially with small or noisy datasets like GTZAN. The faster converge and computational efficient can result in better performance in terms of accuracy, especially for smaller datasets like GTZAN.

### Data Augmentation
We attempted to incorporate data augmentation as a method to introduce variability to the spectrogram data and thereby enhance model generalisation. RandomTranslation(0.1, 0.1) simulates slight shifts in time and frequency, while RandomZoom(0.1) applies small zooming effects, altering the spectrogram's scale to make the model robust to frequency distortions. Gaussian Noise adds random noise with a mean of 0 and a small standard deviation (0.01), preventing over-reliance on specific music features. Random Contrast (0.8, 1.2) varies the contrast of the spectrogram. Overall, these methods are meant to mimick real-world variations in musical recordings and thus help the model becomre more robust, reducing the risk of overfitting.
Unfortunately, data augmentation was giving worse accuracy results for all models that I have tried, other than CNN + GRU which is presented below. This could be due to overly distorted features or insufficient training for the additional variability added in the data.

## Non-data Augmented CNN + GRU: BEST MODEL -> VALIDATION ACCURACY: 86-88.5%
