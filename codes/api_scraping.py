#! python3

import json

def get_keys(path):
    with open(path) as f:
        return json.load(f)

import praw
import pandas as pd
import datetime as dt

keys = get_keys("C:/Users/gupta/.secret/reddit_api.json")
client_id = keys['client_id']
client_secret = keys['api_key']
user_agent = keys['user agent']
username = keys['username']
password = keys['password']

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent = user_agent,
                     username = username,
                     password = password)

subreddit = reddit.subreddit('AskWomen')

# Subreddits
hot_subreddit = subreddit.hot(limit=1000)
new_subreddit = subreddit.new(limit=1000)
controversial_subreddit = subreddit.controversial(limit=1000)
top_subreddit = subreddit.top(limit=1000)
gilded_subreddit = subreddit.gilded(limit=1000)


questions_dict = {"Questions":[]}

# Scraping from 5 subreddits 
for submission in hot_subreddit:
    questions_dict["Questions"].append(submission.title)
for submission in new_subreddit:
    questions_dict["Questions"].append(submission.title)    
for submission in controversial_subreddit:
    questions_dict["Questions"].append(submission.title)    
for submission in top_subreddit:
    questions_dict["Questions"].append(submission.title)    
for submission in gilded_subreddit:
    questions_dict["Questions"].append(submission.title)

# To check uniqueness
questions = questions_dict["Questions"]
len(questions)
len(set(questions))

questions = list(set(questions))

# To convert to csv
questions = pd.DataFrame(questions)

questions.to_csv('data_from_api_main.csv', index=False)
