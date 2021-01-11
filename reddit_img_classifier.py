import logging
import praw
from skimage import io
import cv2
import img_compare

# Logging setup
logging.basicConfig(
    format='''%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] %(message)s''',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('prawcore').setLevel(logging.WARNING)
logging.getLogger('skimage').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# To use this program, please input your own client_id and client_secret for
# the reddit API, either here or in a praw.ini file.
reddit = praw.Reddit('reddit_img_classifier',
    user_agent='meme_classifier: by u/Puzzled_Yellow')

# Works best on subreddits dedicated to memes or just image posts.
def get_top_images(subreddit='all', time='month', post_num=20):
    '''
    Returns a list of cv2 images(numpy arrays) from the specified subreddit's
    top x posts (where x=post_num) for the specified time period.
    '''
    logging.info('Acquiring %s images from %s for the last %s',
        post_num, subreddit, time)
    subreddit = reddit.subreddit(subreddit)
    iter_submission = subreddit.top(time)
    images = []
    while len(images) < post_num:
        try:
            submission = next(iter_submission)
        except StopIteration:
            logging.info('No more submissions to search')
            logging.info('Images acquired: %s', len(images))
            break
        url = submission.url
        if url.endswith('jpg') or url.endswith('jpeg') or url.endswith('png'):
            logging.debug('Acquiring image %s from %s', len(images) + 1, url)
            image = io.imread(url)  # Reads as RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # All images in img_compare assume BGR
            images.append(image)
    logging.info('Done acquiring images')
    return images

def main():
    images = get_top_images('ProgrammerHumor')
    # img_compare.show_images(images)
    img_compare.classifier_func(images, visualize=True)

if __name__ == '__main__':
    main()