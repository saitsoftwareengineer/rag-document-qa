# Recommendation Features

The recommendation pipeline builds features for every user and product
pair from view, cart and purchase events. User features include total
spend, purchase rate and recency. Product features include conversion
rate, revenue and average rating. Pair features include how often the
user viewed or bought that product and whether the product category
matches the user's preferred category. Features are stored in
PostgreSQL for the ranking model.
