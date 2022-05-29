import twint
import time


# ONLY WORKING FOR TWEETS AFTER 2020

c = twint.Config()

c.Search = "Bitcoin OR bitcoin OR BTC OR btc OR #BTC OR $BTC"
c.Custom["tweet"] = [
    "created_at",
    "username",
    "replies_count",
    "retweets_count",
    "likes_count",
    "retweet",
    "tweet",
]

c.Lang = "en"
c.Min_replies = 1
c.Min_retweets = 1
c.Min_likes = 1
c.Since = "2018-01-01"
c.Until = "2023-01-01"

# c.Members_list = "topnewsbitcoin/bitcoin"

# c.Limit = 200000
# c.Hide_output = True
c.Count = True
# c.Stats = True
c.Store_csv = True
c.Output = "bitcoin_scraped.csv"
c.Resume = "my_search_id.txt"


while True:
    twint.run.Search(c)
    print("\n\n starting again... \n\n")
    time.sleep(5)
