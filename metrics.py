from sklearn.metrics import accuracy_score

def evaluate_model(predictions, labels):
    return accuracy_score(labels, predictions)