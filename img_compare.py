import os
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import cv2

TEST_IMG_PATH = 'test_images/'
MSE_CUTOFF = 0.1  # Lower MSE is more similar 
SSIM_CUTOFF = 0.6  # Higher SSIM is more similar
HASH_CUTOFF = 30. # Lower hash difference is more similar

def mse(imgA, imgB):
    '''Returns the Mean Squared Error (MSE) between the two images.'''
    assert imgA.shape == imgB.shape
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err


# Taken from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
# with some modifications
def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def img_folder_plot(path_img_folder):
    '''Plots all the images in a folder and returns a list of image paths.'''
    image_names = os.listdir(path_img_folder)
    image_names.sort()
    images = []
    image_paths = []
    for image_name in image_names:
        image_path = os.path.join(path_img_folder, image_name)
        image_paths.append(image_path)
        images.append(cv2.imread(image_path))
    show_images(images, titles=image_paths)
    return image_paths


def compare_images(imgA, imgB, visualize=False):
    '''Returns a tuple consisting of (MSE, SSIM) of the two images.'''
    if visualize:
        show_images([imgA, imgB])
    hashA = imagehash.average_hash(Image.fromarray(imgA))
    hashB = imagehash.average_hash(Image.fromarray(imgB))
    hash_diff = abs(hashA - hashB)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    if imgA.shape[0] != imgB.shape[0] or imgA.shape[1] != imgB.shape[1]:
        # Note: SSIM is better stretching the smaller image vs. shrinking larger
        max_shape = (max(imgA.shape[0], imgB.shape[0]),
                     max(imgA.shape[1], imgB.shape[1]))
        imgA = resize(imgA, max_shape, anti_aliasing=True)
        imgB = resize(imgB, max_shape, anti_aliasing=True)
    similarity_score = (mse(imgA, imgB), ssim(imgA, imgB), hash_diff)
    if visualize:
        print(f'MSE: {similarity_score[0]}')
        print(f'SSIM: {similarity_score[1]}')
        print(f'HASH: {similarity_score[2]}')
    return similarity_score


def similarity_func(imgA, imgB, \
        MSE_cutoff=MSE_CUTOFF, SSIM_cutoff=SSIM_CUTOFF, HASH_cutoff=HASH_CUTOFF):
    '''Returns True or False if two images are deemed similar.'''
    similarity_score = compare_images(imgA, imgB)
    if similarity_score[0] < MSE_CUTOFF or similarity_score[1] > SSIM_CUTOFF or\
        similarity_score[2] < HASH_CUTOFF:
        return True
    return False


test_img_path = lambda img_name: os.path.join(TEST_IMG_PATH, img_name)

def main():
    pass


if __name__ == '__main__':
    main()
