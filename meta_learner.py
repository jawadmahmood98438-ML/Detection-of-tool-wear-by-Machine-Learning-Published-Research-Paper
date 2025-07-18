import torch

def predict_class(encoder, classifier, query, support_set):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        query_feat = encoder(query).squeeze()
        max_score = -1
        predicted_label = None
        for label, support_img in support_set.items():
            support_feat = encoder(support_img).squeeze()
            score = classifier(query_feat, support_feat).item()
            if score > max_score:
                max_score = score
                predicted_label = label
    return predicted_label