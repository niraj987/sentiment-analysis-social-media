"""
generate_dataset.py
────────────────────
Generate a synthetic dataset of ~1200 product/food reviews
mimicking Amazon, Zomato, and Swiggy review styles.

Run:  python data/raw/generate_dataset.py
Output: data/raw/reviews.csv
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Review templates ──────────────────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "Absolutely loved this product! Exceeded all my expectations.",
    "Amazing quality for the price. Will definitely buy again!",
    "Super fast delivery and the item is exactly as described.",
    "5 stars! This is the best purchase I've made this year.",
    "Outstanding product. Works perfectly and looks great.",
    "The food was absolutely delicious, fresh and hot on arrival!",
    "Best biryani I've ever had. Highly recommend this restaurant.",
    "Delivery was super quick and the packaging was excellent.",
    "Great taste and generous portions. Will order again!",
    "Loved the food quality. Perfect spice level and very fresh.",
    "Fantastic service! The delivery executive was very polite.",
    "Good product, solid build quality, very happy with this purchase.",
    "Excellent experience overall. The app was smooth and order arrived on time.",
    "Very impressed with the quality. Packaging was also very good.",
    "Tastes just like homemade. So glad I found this place!",
    "Wow, just wow! The product blew my mind completely.",
    "Received the package on time. Very satisfied with the seller.",
    "Great value for money. Works as advertised.",
    "The chicken was perfectly cooked, juicy and full of flavor.",
    "No complaints at all. Everything was perfect from order to delivery.",
    "10/10 product. Would strongly recommend to everyone.",
    "Very happy with this purchase. The quality is top-notch.",
    "Crispy, hot and fresh! The fries were amazing.",
    "Love the flavor! This has become my go-to comfort food.",
    "Item was well packed and exactly as shown in the listing.",
]

NEGATIVE_REVIEWS = [
    "Terrible product. Stopped working after just 2 days of use.",
    "Complete waste of money. The quality is very poor.",
    "Very disappointed. The product looks nothing like the photos.",
    "Worst purchase ever. Do not buy this garbage.",
    "Received a damaged item and customer support was useless.",
    "The food was cold and stale when it arrived. Totally unacceptable.",
    "The delivery took over 2 hours and the food was ice cold.",
    "Horrible taste. Nothing like what was shown in the pictures.",
    "Found a hair in my food. Absolutely disgusting!",
    "The order was completely wrong. Got totally different food.",
    "Packaging was broken and everything was spilled. Very bad!",
    "Worst pizza I've ever had. Dough was raw from the inside.",
    "The product broke on the very first use. Pathetic quality.",
    "Seller is a fraud. Sent a fake product instead of the original.",
    "Disgusting! The food smelled awful and tasted even worse.",
    "The movie, the app, the everything — all terrible experience.",
    "Customer service is non-existent. Absolute zero support.",
    "Never received my order. And no refund either. Scammers!",
    "Way overpriced for the terrible quality they delivered.",
    "The food was burnt on one side and raw on the other. Shocking!",
    "Returned the product but didn't get any refund for weeks.",
    "Delivery partner was rude and dropped the bag from far.",
    "Extremely salty food. Couldn't eat more than one bite.",
    "Product is a complete scam. Nothing works as advertised.",
    "Arrived 90 minutes late. By then the food was in a terrible state.",
]

NEUTRAL_REVIEWS = [
    "It's okay. Nothing special but does the job.",
    "Average product. Neither good nor bad. Just decent.",
    "Delivery was on time. Food was so-so, not great.",
    "Would have been better with slightly better packaging.",
    "It's alright. Not as expected but manageable.",
    "The product works fine but could be improved in some areas.",
    "Food was average. Some items were tasty, some were not.",
    "Decent for the price but room for improvement.",
    "Nothing extraordinary. Just a regular experience.",
    "The flavors were okay but nothing to rave about.",
    "Works as described but the build feel is a bit cheap.",
    "Mediocre experience. Expected more for this price.",
    "Food quantity was okay but the quality could be better.",
    "Product is functional but not premium by any means.",
    "Acceptable but I've had better from other sellers.",
    "The taste was fine but the presentation was disappointing.",
    "Order was slightly delayed but food was alright.",
    "Mixed experience. Some things were good, some were not.",
    "Not impressed but not terrible either. Just average.",
    "Got what I paid for. Quality matches the price I guess.",
    "Product arrived in decent shape. Does what it says.",
    "Middle of the road experience. Nothing special happened.",
    "Food was warm enough. Taste was neither good nor bad.",
    "The service was fine. The food was neither special nor bad.",
    "Decent product overall. Would maybe try alternatives next time.",
]

PLATFORMS = ["Amazon", "Zomato", "Swiggy"]
CATEGORIES_AMAZON = ["Electronics", "Clothing", "Home Appliances", "Books", "Beauty", "Sports"]
CATEGORIES_FOOD = ["Indian", "Chinese", "Pizza", "Biryani", "Burgers", "Desserts", "Continental"]

SENTIMENT_MAP = {"positive": 1, "negative": 0, "neutral": 2}

def random_date(start_year=2022, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

def generate_reviews(n=1200):
    records = []
    for i in range(n):
        sentiment_label = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.45, 0.35, 0.20]
        )[0]

        if sentiment_label == "positive":
            text = random.choice(POSITIVE_REVIEWS)
            rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
        elif sentiment_label == "negative":
            text = random.choice(NEGATIVE_REVIEWS)
            rating = random.choices([1, 2], weights=[0.6, 0.4])[0]
        else:
            text = random.choice(NEUTRAL_REVIEWS)
            rating = random.choices([2, 3, 4], weights=[0.2, 0.6, 0.2])[0]

        platform = random.choice(PLATFORMS)
        if platform == "Amazon":
            category = random.choice(CATEGORIES_AMAZON)
        else:
            category = random.choice(CATEGORIES_FOOD)

        # Add slight variations to make reviews more unique
        suffixes = [
            "", " Overall a good choice.", " Highly recommend!",
            " Not worth the money at all.", " Will update after more use.",
            " Ordered twice already.", " My family loves it.",
            " Very disappointed with this.", " Could be much better.",
        ]
        text = text + random.choice(suffixes)

        records.append({
            "review_id": f"REV{i+1:05d}",
            "platform": platform,
            "category": category,
            "rating": rating,
            "sentiment": sentiment_label,
            "date": random_date(),
            "review_text": text,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    import os
    output_path = os.path.join(os.path.dirname(__file__), "reviews.csv")
    df = generate_reviews(1200)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset generated: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nPlatform distribution:\n{df['platform'].value_counts()}")
