# This is the README for OutfitSuggester
### Contributor(s): Jack Burgdoerfer
### Technologies used:
+ Python
+ MongoDB
+ TensorFlow
+ Keras
+ NumPy
+ Sci-Kit Learn
+ Selenium
<br/>
<br/>

# About:
    - Program that was developed over the course of three weeks.
    - Achieved 74% validation accuracy using a Convolutional-Neural-Net.
    - Picks fashionable outfits using clothing that the user owns.

<br/>

# Next Steps:
    - Adding documentation to every method and class.
    - Putting the program in app form using Kivy. 
    - Adding unit, integration and system tests. 
    - Adding a class to help capture user photo input


<br/>

# How it works:

### Data:
    The data was collected using the Selenium API. Over 10,000 
    images were scraped from fashion websites. This can be found in
    Scraper.py. 

    The data was then sanitized and put into proper form. This process
    can be found in FormatPhoto.py. This works by segmenting the background 
    of the input image. The image is then stitched into a single square which 
    features a snippet of the shirt and a snippet of the pants.


### NeuralNetwork.py

    The heart of the program is a Convultional Neural Network
    which can be found in NeuralNet.py. The CNN works using transfer
    learning with ResNet152 as the base layer. The Network then uses many
    residual and dense layers to conclude processing. The Network makes use 
    of a couple of callbacks: early stopping and a learning rate stopper. 
    
    UnsupervisedClustering.py is a class that is used to extract features
    and cluster the data.


### DB.py

    This class is built using MongoDB and is used to insert photos,
    paths to files and collectives that make up an "outfit".

### OutfitGenerator.py

    This class generates all possible combinations of outfits.
    It then feeds these generated outfits into the neural network to predict the liklihood
    of fashionability before adding the outfit to the database. This class is also
    used to get an outfit each day.


### Other
    -SMTP.py is a class that can send emails. Will be used to notify a user of
        important information.

    -Weather.py is a class that gets the weather for the user's current location.

    -Geo.py is a class to get current location.

    -Authentication.py is a class that will be used to authenticate a user's sign in. 
    
    -LocalOutfits.py is a class called by OutfitGenerator.py to help generate outfits.
    
    -Directories.py is a class that holds all of the path directories that are commonly used.
    