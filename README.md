# Introduction

This is a very brief look at the reddit-img-similarity-classifier. For a better understanding and more detail, look in the showcase folder for a more detailed README.md that was made from the Jupyter Notebook file. Better yet, open up the Jupyter Notebook.

This project was initially an attempt to classify meme formats together, later evolving to mainly just classifying similar images together (which includes meme formats).

The img_compare.py file contains all the image comparison and classification stuff. The reddit_img_classifier.py contains the interaction with the (Reddit API (PRAW))[https://praw.readthedocs.io/en/latest/index.html]. 

Obviously, my client_id and client_secret for this project are hidden in a praw.ini I did not copy over. For information on how to use praw.ini, see [here](https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html)

# How it works

The img_compare.py file contains all the dealings of comparing and grouping images.

It uses OpenCV, numpy, matplotlib, scikit-image and more to assist in this process.

Essentially, for any two images (using cv2.imread (BGR) or any other imread that converts to a numpy array (likely RGB)) a similarity score can be calculated. That similarity score is composed of the mean squared error (MSE), the structural similarity index (SSIM), and the hash difference. See the showcase for more details.

A similarity score (or two images) can be inputted into a det_similarity function that determines similarity through what is currently a VERY basic algorithm.

The classifier function takes in a list of images (as numpy arrays) and for each image, creates a category and searches for similar images to also add to said category. If it finds a similar image, the function adds it to the category and deletes the similar image from the image list. The classification function iterates through the image_list in this way until no more images are left to be classified.

The reddit_img_classifier.py file just contains interaction between PRAW and the tools seen in img_compare.py

Using PRAW, it gets the top X amount of posts from a chosen subreddit and categorizes them.

# Improvements

* Improve the det_similarity function to more accurately capture similar images.

* Optimize speed. Right now, this code runs VERY slowly. It's (if I have my big O's correct, which I might not) around O(n^3). Larger batches take very long to process. There are points where optimization is clearly possible.
