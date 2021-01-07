```python
import os
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import cv2
```


```python
from img_compare import *
from reddit_img_classifier import *
```


```python
logging.getLogger('root').setLevel(logging.WARNING)

# Note: modify this to get detailed logging messages to see progress
```

# Introduction

This is a showcase of some image comparison functions and how I use them to try and group Reddit memes together to identify popular meme formats.

I will be using three different image comparison metrics to determine similarity: Mean Squared Error (MSE), Structural Similarity Measure (SSIM), and image hashing.

### Mean Squared Error (MSE)

The Mean Squared Error (MSE) between the two images is the sum of the squared difference between the two images.

It's pretty simple to implement, but it is a decent measure of similarity. It only functions on images of the same size.

MSE has some limitations. For example, large distances between pixel intensities, such as contrasted images, do not necessarily mean the image is not similar. Also, MSE only works on images of the same size.

Note that lower MSEs mean more similar images.

### Structural Similarity Measure (SSIM)

The Structural Similarity Measure (SSIM). SSIM attempts to model the perceived change in the structural information of the image.

Essentially, the main benefit of SSIM is that it accounts for texture. It measures similarity, just as MSE, but takes into account image quality.

SSIM is likely the strictest of the three measures used to indicate image similarity.

SSIM also only works on images of the same size. From testing, SSIM seems to work better when smaller images are resized to fit the larger than the opposite direction (larger images being resized to fit smaller).

Note that higher SSIMs reflect more similarity.

The SSIM function used for image comparison is imported from the [scikit-image library](https://scikit-image.org/)

### Image Hashing

Image hashing or perceptual hashing is the process of examining the contents of an image and then constructing a hash value that uniquely identifies an input image based on the contents of the image.

Essentially, it narrows the image down to a set of unique features and assigns it a value based on those features.

The image hashes of two similar images will be more similar than the image hashes of two different images. Therefore, by subtracting the image hashes, one can get a rough measure of how similar two images are.

Image hashing works regardless of image size.

Note that a lower hash difference means the images are more similar.

# Examples

Let's see some example images and how they compare using our similarity measures.

Run the following block to see all available test images:


```python
test_image_paths = img_folder_plot(TEST_IMG_PATH)
imgA = cv2.imread(test_image_paths[0])
imgB = cv2.imread(test_image_paths[1])
imgC = cv2.imread(test_image_paths[2])
```


    
![png](output_8_0.png)
    


Let's compare all the various images now. As can be seen, spongebob1.png and spongebob2.png are the same meme format, so they should be more similar. spongebob3.png should have a notably worse similarity score.


```python
compare_all_images([imgA, imgB, imgC], visualize=True)
```


    
![png](output_10_0.png)
    


    MSE: 0.07560973075958617
    SSIM: 0.6437364351044425
    HASH: 23



    
![png](output_10_2.png)
    


    MSE: 0.23747421165086122
    SSIM: 0.4032295038558613
    HASH: 32



    
![png](output_10_4.png)
    


    MSE: 0.228390133086145
    SSIM: 0.35620894495679245
    HASH: 35


As can be seen the two images with the same meme format have lower MSEs, higher SSIMs, and lower hash differences than images with different meme formats.

Although, the scores may still be a bit off from what you may have expected.

For example, why do the images with the same meme format still have such a low SSIM and such a high hash?

It's important to remember SSIM measures texture, making it perhaps the strictest of the three measures. Memes come in many different qualities, so SSIM, while still showing similarity, will be harsher towards different quality images.

Hashes are used to compare when images are nearly identical. Not just the different quality but the different text affects how well hashes work, although of course they still reflect the similarity of the two images.

SSIM and Hashes are both stricter metrics than MSE. Although, they are both still important. For example, SSIM will be more resistant to contrasting a meme format (a common practice) relative to MSE.

# Basic Classification

Given any one of MSE, SSIM, or hash difference, it would not be too difficult to write a simple function to determine similarity.

The code in img_compare.py contains a det_similar function that uses all three (MSE, SSIM, hash_diff) as a tuple called a "similarity_score" throughout the code (although I didn't bother with OOP for this).

det_similar accepts either two images or a similarity score, and returns a boolean. 

Currently, it's rather naive, and calls images similar if any of the metrics qualifies for similarity. 

In the below code block for your convenience is the det_similar function. It's identical to the one in the code, so running it shouldn't change anything.


```python
def det_similar(imgA=None, imgB=None, similarity_score=None, \
        MSE_cutoff=MSE_CUTOFF, SSIM_cutoff=SSIM_CUTOFF, HASH_cutoff=HASH_CUTOFF):
    '''Returns True or False if two images are deemed similar.'''
    assert imgA and imgB and not similarity_score \
            or similarity_score and not imgA and not imgB
    if not similarity_score and imgA and imgB:
        similarity_score = compare_images(imgA, imgB)
    if similarity_score[0] < MSE_cutoff or similarity_score[1] > SSIM_cutoff or\
        similarity_score[2] < HASH_cutoff:
        return True
    return False
```

Classifying isn't too hard with this function to determine similarity. My function simply assigns the first image in a list as the first image in the category. Then it iterates through the rest of the list to determine similarity. If similarity is found, it appends the image to the category and removes it from the image list.

## Example

Let's see my basic similarity determination and classification in action with the test images from above.


```python
classifier_func([imgA, imgB, imgC], visualize=True);
```

    Category 1:



    
![png](output_16_1.png)
    


    Category 2:



    
![png](output_16_3.png)
    


As can be seen, this classification works for the test images. Although, there are only three of them, and they are pretty clearly distinct. Let's try with some more images.

# Classifying Reddit Posts

The comparisons above can also be used to classify image posts on popular media aggregation site Reddit. In particular, the image posts we will be classifying will all be (or all should be) memes.

Memes are hopefully much easier to classify than other types of images because they mainly all follow a format. For example, the two different 2020 Spongebob memes above both use the same meme format. This makes it easy to categorize them by similarity.

## The Python Reddit API Wrapper (PRAW)

Luckily, Reddit has an API to be able to easily collect information about posts. For Python, this is the Python Reddit API Wrapper (PRAW).

The code block below gets the top post_num image posts from the subreddit r/ProgrammerHumor from the past month. It shows only the first 20.

If you're viewing this on Jupyter Notebook, you can edit these values. I advise picking a subreddit that's based solely on memes.


```python
images = get_top_images(
    subreddit='ProgrammerHumor',
    time='month',
    post_num=50
)
show_images(images, show=20)
```


    
![png](output_21_0.png)
    


Let's check for any similar images, ideally meme formats, using our classifer.

WARNING: there are lots of different categories. Be prepared to scroll.


```python
classifier_func(images, visualize=True);
```

    Category 1:



    
![png](output_23_1.png)
    


    Category 2:



    
![png](output_23_3.png)
    


    Category 3:



    
![png](output_23_5.png)
    


    Category 4:



    
![png](output_23_7.png)
    


    Category 5:



    
![png](output_23_9.png)
    


    Category 6:



    
![png](output_23_11.png)
    


    Category 7:



    
![png](output_23_13.png)
    


    Category 8:



    
![png](output_23_15.png)
    


    Category 9:



    
![png](output_23_17.png)
    


    Category 10:



    
![png](output_23_19.png)
    


    Category 11:



    
![png](output_23_21.png)
    


    Category 12:



    
![png](output_23_23.png)
    


    Category 13:



    
![png](output_23_25.png)
    


    Category 14:



    
![png](output_23_27.png)
    


    Category 15:



    
![png](output_23_29.png)
    


    Category 16:



    
![png](output_23_31.png)
    


    Category 17:



    
![png](output_23_33.png)
    


    Category 18:



    
![png](output_23_35.png)
    


    Category 19:



    
![png](output_23_37.png)
    


    Category 20:



    
![png](output_23_39.png)
    


    Category 21:



    
![png](output_23_41.png)
    


    Category 22:



    
![png](output_23_43.png)
    


    Category 23:



    
![png](output_23_45.png)
    


    Category 24:



    
![png](output_23_47.png)
    


    Category 25:



    
![png](output_23_49.png)
    


    Category 26:



    
![png](output_23_51.png)
    


    Category 27:



    
![png](output_23_53.png)
    


    Category 28:



    
![png](output_23_55.png)
    


    Category 29:



    
![png](output_23_57.png)
    


    Category 30:



    
![png](output_23_59.png)
    


    Category 31:



    
![png](output_23_61.png)
    


    Category 32:



    
![png](output_23_63.png)
    


    Category 33:



    
![png](output_23_65.png)
    


    Category 34:



    
![png](output_23_67.png)
    


    Category 35:



    
![png](output_23_69.png)
    


    Category 36:



    
![png](output_23_71.png)
    


    Category 37:



    
![png](output_23_73.png)
    


    Category 38:



    
![png](output_23_75.png)
    


    Category 39:



    
![png](output_23_77.png)
    


    Category 40:



    
![png](output_23_79.png)
    


As can be seen, the classifier does classify similar images. Although, those similar images aren't necessarily memes as I would've hoped.

For example, many Twitter posts (a common post on r/ProgrammerHumor and many other subreddits) were classified the same since they're generally very similar images. Also, many photos that are mainly just black text on a white background suffered the same fate.

Ideally, the det_similarity function will be more strict with what it considers similar.

Another problem is that there are surprisingly few similar meme formats. This is likely because images are taken from the top posts and not the hot or new posts, where similar meme formats are more likely. Alternatively, there are also so many meme formats that it would be more accurate to compile thousands of posts than just the few I did. I didn't want to take too large of a batch for a showcase though.

Overall, while the program failed to classify meme formats in this example, it did classify similar images and showed potential to classify meme formats given more data and a better subreddit.

# Improvements

* Improve the det_similarity function to more accurately capture similar images.


* Optimize speed. Right now, this code runs VERY slowly it's (if I have my big O's correct, which I might not) O(n^3). Larger batches take very long to process. There are points where optimization is clearly possible here.
