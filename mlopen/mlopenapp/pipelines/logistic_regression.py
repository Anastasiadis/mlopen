from sklearn.linear_model import LogisticRegression

def log_reg(x, y):
    print(len(x))
    print(len(y))
    sentiment_model = LogisticRegression()
    sentiment_model.fit(x, y)
    return sentiment_model
