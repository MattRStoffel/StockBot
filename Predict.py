import torch
from Data import text_pipeline

label_pipeline = lambda x: int(x) - 1

model = torch.load("model.h1")

def predict(text, text_pipeline):
    with torch.no_grad(): # disables gradient calculation
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

#Test on a random news
news_label = {1: "berish", 2: "bullish"}

ex_text_str = """
I am bullish
"""

model = model.to("cpu")

print("This is a %s news" % news_label[predict(ex_text_str, text_pipeline)])