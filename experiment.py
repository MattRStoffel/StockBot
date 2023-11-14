# import torch
from torch import tensor, load, no_grad, save
from data import data
from sentiment_data import sentiment_data 
from train import trainer
from pipeline import pipeline
from TextClassificationModel import TextClassificationModel as model

pl = pipeline(sentiment_data())
data = data(sentiment_data(), pl)
number_of_classes = len(set([label for (label, text) in sentiment_data()]))
vocab_size = len(pl.vocab)
embedding_size = 64

model = model(vocab_size, embedding_size, number_of_classes).to("cpu")

trainer = trainer(data, model, 9, 5)
trainer.run()
model = trainer.model
save(model, "model.h9")

def predict(text):
    with no_grad(): # disables gradient calculation
        text = tensor(pl.text_pipeline(text))
        output = model(text, tensor([0]))
        return (output.argmax(1).item() + 1, output.tolist())

test_tests = [
    "BAD",
    "GOOD",
    "The markets are looking terrible today",
    "I hate myself",
    "Life is good",
    "Buy Buy Buy",
    "Good move or bad move? $BRY Aroon Indicator entered a Downtrend",
    "What are the Hedge Funds going to do now? $AVNW RSI Indicator left the overbought zone"
]

model = load("model.h9")

labels = sentiment_data().labels
for test in test_tests:
   p = predict(test)
   print("| {:<40} | {:<20} | {}".format(test[:40],labels[p[0]],p[1]))
