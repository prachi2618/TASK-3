# Image Classifier using Convolutional Neural Networks (CNN)

This project uses TensorFlow and Keras to build and train a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset.

## Project Overview

The goal of this project is to build an image classifier capable of recognizing 10 different classes of images using the CIFAR-10 dataset. The dataset contains 60,000 32x32 color images in 10 categories, with 6,000 images per category.

## Prerequisites

Before running the project, make sure you have the following libraries installed:

- `tensorflow`
- `numpy`
- `matplotlib`
- `keras`

You can install them using `pip`:

```bash
pip install tensorflow numpy matplotlib
```

## Dataset

The CIFAR-10 dataset contains the following 10 classes:
- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is automatically downloaded from the TensorFlow `keras.datasets` module.

## Code Explanation

1. **Loading the CIFAR-10 Dataset**:
   The CIFAR-10 dataset is loaded using TensorFlow's `cifar10.load_data()` function, which returns the training and testing images and labels.

2. **Data Preprocessing**:
   The pixel values of the images are normalized to a range between 0 and 1 by dividing each pixel value by 255.0.

3. **Visualizing the Data**:
   A set of 16 images from the training dataset is displayed using `matplotlib`.

4. **Model Architecture**:
   A Convolutional Neural Network (CNN) is defined using the following layers:
   - 3 Conv2D layers (with ReLU activation) followed by MaxPooling2D layers for downsampling.
   - A Flatten layer to reshape the output of the convolutional layers.
   - Two Dense layers: one hidden layer with 64 units (ReLU activation) and one output layer with 10 units (softmax activation).

5. **Model Compilation**:
   The model is compiled using the Adam optimizer and sparse categorical crossentropy loss function for multi-class classification.

6. **Training the Model**:
   The model is trained for 10 epochs using the training data and evaluated using the test data.

7. **Model Evaluation**:
   After training, the model is evaluated on the test set to calculate the loss and accuracy.

8. **Saving the Model**:
   The trained model is saved as `image_classifier.h5`.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/cifar-10-cnn-image-classifier.git
cd cifar-10-cnn-image-classifier
```

2. Run the script to train the model:

```bash
python image_classifier.py
```

3. The trained model will be saved as `image_classifier.h5` in the project directory.

## Example Output

After training for 10 epochs, you will see the loss and accuracy on the test dataset:

```
Loss: <value>
Accuracy: <value>
```

## Future Improvements

- Increase the number of epochs to improve model performance.
- Add data augmentation to artificially increase the size of the dataset.
- Experiment with other architectures like ResNet, VGG, etc., to improve accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Notes:
- Replace `https://github.com/yourusername/cifar-10-cnn-image-classifier.git` with the actual repository URL if you're hosting this project on GitHub.
- Customize any additional details you wish, like the usage, future improvements, or specific instructions for your project setup.
