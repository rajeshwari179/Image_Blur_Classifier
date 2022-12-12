# Image Classifier into Blurred and Not Blurred

Here I am built a multi-model classifier which identifies a given image as Blurred or Not blurred!

The dataset used for this classifier is the [CERTH Image Blur dataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip)

The dataset has a training set, which has 150 artificially blurred images, 220 naturally blurred images and 630 undistorted images.

It also has an evaluation set.

Note: TL;DR and link to the Zip file at the end.

## Model Description

I have trained a `Convolutional Neural Network` model on the training set. The CNN has 4 Convolutional layers followed by 2 dense layers. The first Conv. layer uses 2 Conv layers with `32 filters`, followed by `Max Pooling`. The second Conv. layer has `64 filters` followed by `Max Pooling`. The last Conv. layer uses `128 filters` and is also followed by max pooling. The dense layers have `2048 inputs` each and have a `Dropout` of `0.5` added to it to prevent `overfitting`. The it passes through a SoftMax classifier, to give the final outcome.

The model was trained on the dataset and demonstrated an accuracy of `84.59%` on the test set.

There is another model based on `OpenCV` that uses the variance of the `Laplacian Kernel` applied to the image. The higher the variance, the lower is the blurriness of the image. This is because Laplacian Kernel identifies edges.

To further improve the accuracy of our model, these outputs of the CNN were combined with the outputs of the OpenCV model and were fed to another `Deep Neural Network` with 5 layers and about `4096 to 2048 inputs` in each layer. This combined model predicts whether the image is blurry or not with an accuracy of `86.55%` on the test set.

### Getting the dependencies

To download all the dependencies, I have included the `requirements.txt` file. To run it, enter:
> pip install -r requirements.txt

## How to run this code

This code has been divided into multiple files to ease the process of training and evaluation. First we need to run the file `load_traindata.py` and `load_testdata.py`.

These files use the raw images and convert them into PKL files (which is just serialised files).

>python load_traindata.py
>python load_testdata.py

### Some common troubleshooting:

The data set needs to be in this file directory format: <br>
\>CERTH_ImageBlurDataset/\<the_actual_dataset\><br>

Note: The zip already includes the files in the right directory order.

The above 2 python programs generate 4 files, X_test.pkl, X_train.pkl, y_test.pkl, y_train.pkl.

These files have been already zipped.

After that, we need to run the `CNN.py` file. This file trains the CNN model and saves the weights so that we can use it to train the Deep neural network for the final prediction.

The CNN was trained on Google Colaboratory using GPU acceleration turned on. The training takes about 2secs per epoch. It was trained for 10 epochs.

To run the code type:
> python `CNN.py`

Note: Run it on Google Colaboratory or other similar service as the model is pretty huge and can easily eat up most of the memory of your personal computers. The model has already been trained and saved in the `model` directory.

Finally we have to run `train.py` to get the final Neural Network trained.

> python `train.py`

Note: Run it on Google Colaboratory or other similar service as the model is pretty huge and can easily eat up most of the memory of your personal computers. The model has already been trained and saved in the `final_model` directory.

### Troubleshooting and clarifications

In case of any errors, check for the file paths in the code. They should be pointing to correct locations. <br>
A very good question would be, why am I training the CNN and DNN separately? Simply put, because I quickly ran out of memory running just the CNN. So I modularised my code into seperate training and testing modules.

## Testing

To evaluate the code on the testing set, you need to run the file `test.py`. It loads the DNN model trained in the last step and loads the y_test and X_test from the file and performs accuracy and loss measurement. <br> Look for the path of the files in case of any errors.

## Final Notes

All the models that are supposed to be loaded have been already included in the ZIP. The model may take time to load or run on local systems so it is preferred that you run this on a cloud based service which provides GPUs for training. `Your mileage might differ`.

### Scope for improvement

This model uses a CNN built from scratch to perform the prediction. The training data was smaller in size than the test data. The training `data was not augmented` in any way. So maybe, data augmentation might be a good way to improve the performance of the model.<br>

Also, there are various pre-existing Image recognition models available to use. So, some form of `Transfer learning` could impact the performance of the model.

#### TLDR

Order of execution:<br>
`load_traindata.py` > `load_testdata.py` > `CNN.py` > `train.py` > `test.py`
<br>
Files generated:<br>
`X_train.pkl` and `y_train.pkl` by `load_traindata.py` <br>
`X_test.pkl` and `y_test.pkl` by `load_testdata.py` <br>
model directory by `CNN.py` <br>
final_model directory by `train.py` <br>

Link to the zip with all the files: [Click here](https://drive.google.com/file/d/10UbDkVdPQOAXhIBukUkcxCe9I0RvK8jg/view?usp=sharing)
