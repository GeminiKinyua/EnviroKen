# EnviroKen
A simple web app, that integrates a waste classification model to help categorize waste images. 
# Technologies Used
Python
Tensorflow
Keras
Pillow
Numpy
Streamlit
# Summary
This project aimed to tackle the environmental issue of improper waste disposal by employing a unique approach that combine technology and behavioural economics. The main deliverable was an image classification model integrated into an app. The model architecture is a VGG16 that was initially trained on the ImageNet Dataset. Transfer learning was used to fine tune the model to learn how to classify waste. The model was then deployed on an interactive web application to allow users classify their own waste images. A reward was then provided depending on the class of the waste item.  
# Dataset
The rich dataset used in this project was sourced from Kaggle, which is a global platform known as the largest community for data scientists and machine learning enthusiasts. Kaggle not only provides numerous datasets  but also hosts competitions that push the frontiers of what’s possible in the realm of data science. The dataset is an extensive collection of over 22500 images compiled by a dedicated individual who compiled images of common everyday waste.
# Model architecture
![vgg16 - architechture](https://github.com/GeminiKinyua/EnviroKen/assets/70109027/1fceb85e-60c2-4c8c-9d64-6cc736982076)

In this case, I use the VGG16 model, which is a convolutional neural network (CNN) model primarily built for object recognition. It was trained on millions of images from the ImageNet database. It is known for its simplicity and high performance, reaching the top accuracies in image classifications tasks. ​

The architecture of VGG16 consists of 16 layers, including 13 convolutional layers and 3 fully connected (FC) layers. However, for this project, I replace the last FC layer of VGG16 with a custom layer to match the number of classes in consideration – Plastic, Glass, Metal, Cardboard, and Paper. ​

To fine tune the model, I froze the initial layers and trained the last few layers on the dataset. This  is because the initial layers of a convolutional neural network learn basic features, and the later layers learn more complex, task-specific features. By freezing the initial layers, basic features learned from ImageNet are preserved, and by training the last few layers, the model learns to recognize waste-related features. ​

# Streamlit App
I am yet to deploy the app on streamlit cloud, but I'm working on that. Users simply upload a waste image, the model then analyses it and predicts the class of the uploaded image. I'm also working on a reward system that will give users a reward depending on the predicted class. This is aimed at encouraging sustainable behaviour through reuse and recycling. 

<img width="636" alt="image" src="https://github.com/GeminiKinyua/EnviroKen/assets/70109027/962b9420-11d4-4253-84c5-ba18c2b48081">

# Future Directions
Develop a stacked model, such that a user can upload a single image with multiple items. The model will first detect the items in the image, then categorize each item accordingly. 
Integrate a sophisticated reward system that takes into consideration the quantity of waste a user is recycling.


