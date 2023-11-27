# Caffe-SSD-Object-Detection
Object Detection using Single Shot MultiBox Detector with Caffe MobileNet on OpenCV in Python.

## SSD Framework
Single Shot MultiBox Detectors can be divided into two parts:
 
 - Extracting Features using a base network
 - Using Convolution Filters to make predictions
 
 This implementation makes use of the MobileNet deep learning CNN architecture as the base network. 

## Caffe Framework
Caffe is a deep learning framework developed by the Berkely AI Research and Community Contributors. It is a much faster way of training images with over 6 million images per day using an Nvidia K-40 GPU

## Run code
`python Object_Detection.py -p MobileNet_prototxt -m MobileNet.caffemodel`


