# Regarding Reddit API
praw can traverse atmost 1000 posts (Reddit's request limit).

To bypass that limit, pushshift can be used. But, AFAIK (after reading the doc - https://github.com/pushshift/api), it appears it can be
used only to search posts with certain keywords, not to traverse over all the posts in a certain time period. 

(Future upgrade of this repo could add what kind of questions men or women ask over a common topic, i.e. the posts with the keyword
'relationships','ex', 'love' could be added searched to make the appropriate dataset using pushshift.)
