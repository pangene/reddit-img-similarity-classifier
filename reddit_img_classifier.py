from img_compare import *
import praw

reddit=praw.Reddit('reddit_img_classifier',
    user_agent='meme_classifier: by u/Puzzled_Yellow')

print(reddit.read_only)
print(reddit.user.me())
