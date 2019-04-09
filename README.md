# fast-style-transfer-tfjs
 
This is based on the following repository https://github.com/misgod/fast-neural-style-keras. I made many changes to make it convertible to TF.js. I only included the files that are important and refactored many functionalities. 

## Pre-requisites

You need keras 2 with tensorflow backend 

## Test Prediction 

`python transform.py -i rose.jpg -s la_muse -b 0.1 -o  out`

This will create the output image and a keras .h5 file 

## Conversion of the models 

`tensorflowjs_converter --input_format keras keras.h5 output/`

## Load the model to the browser

This is done using tensorflow.js check the file fast-style.html. Note that I have and editted version of the source package `tf.min.js`. It containts many custom layers like cropping and upsampling that are yet to be implemented in tf.js. Make sure to use that file.

https://shreyanshp.github.io/fast-style-transfer-tfjs/

![Alt text](screen-shot.png?raw=true "Title")
