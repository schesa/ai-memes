# Ai memes

Work In Progress...

[About](#About) | [Web Arhitecture](#web-arhitecture) | [Neural Network](#neural-network) | [ImgFlip API](#imgflip-api) | [GRPC Server](#grpc-server) | [Backend](#backend) | [Frontend](#frontend) | [Examples](#examples) | [References](#references)

# About

Generating memes using Neural Networks

Dataset used: **[ImgFlip575K_Dataset](https://github.com/schesa/ImgFlip575K_Dataset)**

# Neural Network

Code **->** net/ai-memes.ipynb 

Colab notebook **->** [ai-memes.ipynb](https://colab.research.google.com/drive/1LnE0DmonhHVZ9RsGKUpO7NW0eDdON8sl?usp=sharing)

Used Show and Tell Model[[1]](#1)[[2]](#2).

## Training

### Colab

* Trained for all memes
* 50 epochs, batch size 32
* GPUs: Tesla K80 ..etc

### Windows 10

* Managed to train for a small number of memes(10).

* Used GPU NVIDIA GeForce GTX 950M with : python 3.6.8, CUDA 9.0, CuDNN 7.3.1, TF-GPU 1.12.

# Web Arhitecture

![Arhitecture](https://github.com/schesa/ai-memes/blob/master/Web-Arhitecture-EN.png)


## GRPC Server

## Backend

## Frontend

## ImgFlip API

### To run
```sh
$> cd ./api
```
```sh
$> touch .env
```
Add your ImgFlip account info to .env file
```
IMGFLIP_USERNAME=<your ImgFlip account>
IMGFLIP_PASSWORD=<your ImgFlip password>
```
Run the server
```sh
$> npm start
```

### Api examples

Call examples found in /api/src/index.js

ex1. https://i.imgflip.com/3vh5hr.jpg

ex2. https://i.imgflip.com/3vh5hs.jpg

# Examples

![ex1](https://raw.githubusercontent.com/schesa/ai-memes/master/net/classes/aquarium.jpg)

![ex2](https://raw.githubusercontent.com/schesa/ai-memes/master/net/classes/iphone-meme.jpg)


# References

<a id="1">[1]</a> 
Oriol Vinyals, Alexander Toshev, Samy Bengio, & Dumitru Erhan.
(2014).
Show and Tell: A Neural Image Caption Generator.

<a id="2">[2]</a> 
Jeff Heaton. 
Washington University (in St. Louis) Course T81-558: Applications
of Deep Neural Networks Module 10: Time Series in Keras.
https://github.com/jeffheaton/t81_558_deep_learning

