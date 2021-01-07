import os
import logging
import imagehash
from PIL import Image  # necessary for imagehash
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import cv2  # cv2 may not be necessary, could read using skimage?

# Logging setup
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logging.getLogger('skimage').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


TEST_IMG_PATH = 'test_images/'
MSE_CUTOFF = 0.08  # Lower MSE is more similar
SSIM_CUTOFF = 0.6  # Higher SSIM is more similar
HASH_CUTOFF = 30 # Lower hash difference is more similar


# UTILITY FUNCTIONS

test_img_path = lambda img_name: os.path.join(TEST_IMG_PATH, img_name)


def show_images(images, titles=None, cols=4, show=None):
    '''Display a list of images in a single figure with matplotlib.'''
    if titles:
        assert len(images) == len(titles)
    if show:
        images = images[:show]
    rows = len(images) // cols + 1
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        if titles:
            plt.title(titles[i])
        else:
            plt.title(f'Image {i + 1}')
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()


def img_folder_plot(path_img_folder):
    '''Plots all the images in a folder and returns a list of image paths.'''
    image_names = os.listdir(path_img_folder)
    image_names = filter(lambda x: not x.startswith('.'), image_names)
    image_names = list(image_names)
    image_names.sort()
    images = []
    image_paths = []
    for image_name in image_names:
        image_path = os.path.join(path_img_folder, image_name)
        image_paths.append(image_path)
        images.append(cv2.imread(image_path))
    show_images(images, titles=image_names)
    return image_paths

# COMPARISON STUFF
# TODO: Should this be a class?

def mse(imgA, imgB):
    '''Returns the Mean Squared Error (MSE) between the two images.'''
    assert imgA.shape == imgB.shape
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err


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


def compare_all_images(img_list, visualize=False):
    '''Compares all images in the list.'''
    while img_list:
        imgA = img_list[0]
        for img in img_list[1:]:
            imgB = img
            compare_images(imgA, imgB, visualize=visualize)
        img_list = img_list[1:]


def det_similar(imgA=None, imgB=None, similarity_score=None, \
        MSE_cutoff=MSE_CUTOFF, SSIM_cutoff=SSIM_CUTOFF, HASH_cutoff=HASH_CUTOFF):
    '''Returns True or False if two images are deemed similar.'''
    assert imgA and imgB and not similarity_score \
            or similarity_score and not imgA and not imgB
    if not similarity_score and imgA and imgB:
        similarity_score = compare_images(imgA, imgB)
    if similarity_score[0] < MSE_cutoff and similarity_score[1] > SSIM_cutoff and\
        similarity_score[2] < HASH_cutoff:
        return True
    return False


def classifier_func(img_list, visualize=False, verbose=False):
    '''
    Returns a list of lists, with each nested list being a category of similar
    images.
    '''
    logging.info('Classifying...')
    if not verbose:
        orig_img_list = img_list[:]
    categories = []
    i = 1  # Used for debugging + to help track sim scores
    while img_list:
        logging.info('Grouping new category...')
        imgA = img_list[0]
        category = [imgA]
        other_images = img_list[1:]
        for img in other_images:  # TODO: Some way to track which images were deemd similar from original list.
            imgB = img
            sim_score = compare_images(imgA, imgB)
            if det_similar(similarity_score=sim_score):
                if verbose:
                    logging.info(f'Images were deemed similar')
                else:
                    other_index = None
                    for j, img in enumerate(orig_img_list[i - 1:]):
                        if np.array_equal(img, imgB):
                            other_index = j + i
                            break
                    assert other_index > i
                    logging.info(f'Images {i} and {other_index} were deemed similar')
                logging.debug(f'Similarity_score: {sim_score}')
                category.append(imgB)
                # Note: list.remove fails with a list of np arrays
                index = 1
                for img in img_list[1:]:  # This can be optimized.
                    if np.array_equal(img, imgB):
                        img_list.pop(index)
                        break
                    index += 1
        categories.append(category)
        i += 1
        img_list = img_list[1:]
    logging.info('Done classifying')
    if visualize:
        for i, category in enumerate(categories):
            print(f'Category {i + 1}:')
            show_images(category)
    return categories


def main():
    test_image_names = os.listdir(TEST_IMG_PATH)
    test_image_names = filter(lambda x: not x.startswith('.'), test_image_names)
    test_image_names = list(test_image_names)
    test_image_names.sort()
    test_image_paths = [test_img_path(image_name) for image_name in test_image_names]
    test_images = [cv2.imread(image_path) for image_path in test_image_paths]
    # compare_all_images(test_images, visualize=True)
    classifier_func(test_images, visualize=True)


if __name__ == '__main__':
    main()
