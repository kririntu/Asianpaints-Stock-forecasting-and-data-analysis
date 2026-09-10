import requests
import feedparser
import pandas as pd
import re


queries = [
    "crude oil",
    "inflation",
    "war",
    "Trump tariff",
    "Asian Paints",
    "housing market",
    "economy india",
    "real estate",
    "Covid 19"
]


def clean_text(text):

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = text.lower()

    return text.strip()


bad_words = [
    "live",
    "statista",
    "price annually"
]


def is_good_news(text):

    for word in bad_words:

        if word in text.lower():

            return False

    return True


def collect_news():

    all_news = []

    for query in queries:

        url = (
            "https://news.google.com/rss/search?"
            f"q={query}+when:2000d&hl=en-IN&gl=IN&ceid=IN:en"
        )

        try:

            response = requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            feed = feedparser.parse(response.content)

            for entry in feed.entries:

                all_news.append({

                    "query": query,

                    "title": entry.title,

                    "published": entry.get("published", ""),

                    "link": entry.link
                })

        except Exception as e:

            print(f"Error for {query}: {e}")

    df = pd.DataFrame(all_news)

    df["title"] = df["title"].apply(clean_text)

    df = df[df["title"].apply(is_good_news)]

    df["published"] = pd.to_datetime(df["published"])

    df.sort_values("published", inplace=True)

    return df