# Customer Churn Pipeline

The churn pipeline runs daily. It cleans customer activity data,
engineers features such as tenure, monthly charges, support tickets,
late payments and tickets-per-login, and trains a random forest
classifier. Predictions and churn probabilities are written to a table
for downstream analytics and retention campaigns. Customers on
month-to-month contracts with several late payments and many support
tickets are the most likely to churn.
