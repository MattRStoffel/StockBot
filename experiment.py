# import torch
import torch
from TextDataLoader import TextDataLoader
from SentimentDataSource import SentimentDataSource 
from RedditDataSource import RedditDataSource
from ModelTrainingManager import ModelTrainingManager
from TextPreprocessor import TextPreprocessor
from TextClassificationModel import TextClassificationModel

EPOCHS = 9
LR = 5
BATCH_SIZE = 64

raw_reddit_data = RedditDataSource()

raw_data = SentimentDataSource()
processed_sentiment = TextPreprocessor(raw_data)
processed_reddit = TextPreprocessor(raw_reddit_data)
sentiment_data = TextDataLoader(raw_data, processed_sentiment, BATCH_SIZE)
reddit_data = TextDataLoader(raw_reddit_data, processed_reddit, BATCH_SIZE)

def make_model(epochs, LR, embedding_size, processed_text, data):
    model = TextClassificationModel(len(processed_text.vocab), embedding_size, data.number_of_classes).to("cpu")
    trainer = ModelTrainingManager(model, LR, len(data.train))
    trainer.train_and_evaluate(epochs, data.train, data.valid)
    return trainer.model

def thredded_train(processed_text, data, max_epochs : int = 15, learning_rates : [float]= [5,4], embedding_sizes : [int] = [64, 32]):
    from concurrent.futures import ThreadPoolExecutor
        
    def make_model_wrapper(args):
        EP, L, em = args
        from torch import nn
        layer_lists = [
            nn.ModuleList([
                nn.Linear(em,em)
            ]),
            nn.ModuleList([
                nn.Linear(em,em),
                nn.Linear(em,em)
            ]),
            nn.ModuleList([
                nn.Linear(em, em),
                nn.ReLU(),
                nn.Linear(em, em)
            ])
        ]
        for i, ll in enumerate(layer_lists):
            model = TextClassificationModel(len(processed_text.vocab), em, data.number_of_classes, ll).to("cpu")
            trainer = ModelTrainingManager(model, L, len(data.train))
            for e in range(0, EP + 1):
                trainer.train_and_evaluate(1, data.train, data.valid)
                torch.save(model, "./models/m" + str(i) + '.' + str(e+1) + '.' + str(L) + '.' + str(em))

    parameter_combinations = [(max_epochs, L, em) for L in learning_rates for em in embedding_sizes]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(make_model_wrapper, parameter_combinations)

def evaluate(model : TextClassificationModel):
    trainer : ModelTrainingManager = ModelTrainingManager(model, model.initial_LR)
    print("E : {:<} | LR : {:<} | Embedding_Size {} | {:4f}".format(model.trained_epochs, model.initial_LR, model.embedding_dim, trainer.evaluate_accuracy(data.test)))

def multi_evaluate(data, directory = './models'):
    import os
    import matplotlib.pyplot as plt
    import random
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    models = []
    for i, file in enumerate(files):
        model = torch.load(directory + '/' + file)
        trainer = ModelTrainingManager(model, LR, len(data.train)) 
        accuracy = trainer.evaluate_accuracy(data.test)
        models.append((accuracy, file))
        plt.scatter(i, accuracy)

    plt.ylabel('Test accuracy')
    plt.show()

    models = sorted(models)
    for model in models:
        print("{:13} | {:3f}".format(model[1], model[0]))

def visualize(model, processed_text, raw_data, number_of_points = 0, r = 'r', g = 'g', y = 'y'):    
    import matplotlib.pyplot
    import random
    
    plt = matplotlib.pyplot

    test_tests = raw_data.data
    random.shuffle(test_tests)

    plt.scatter(0,0,color='g')
    plt.scatter(0,0,color='r')
    plt.scatter(0,0,color='y')
    plt.scatter(0,0,color='w')
    plt.legend(["Bullish", "Bearish", "Wrong Guess"])

    for label, text in test_tests[:number_of_points if number_of_points > 0 else len(test_tests)]:
        try:
            predicted_label, tensor_values = model.predict(processed_text, text)
            tensor_values = tensor_values[0]
            largest_value = tensor_values[0] if tensor_values[0] > tensor_values[1] else tensor_values[1]
            tensor_sum = tensor_values[0] + tensor_values[1]
            accuracy = (r if predicted_label == 1 else g) if label != predicted_label+1 else y
            if accuracy == r:
                pass
            plt.scatter(tensor_sum, largest_value, color=accuracy)
        except:
            pass

    plt.xlabel('Tensor sum')
    plt.ylabel('Largest tesnsor value')
    plt.show()
    plt.savefig("plot.x")




# thredded_train(processed_sentiment, sentiment_data)
# multi_evaluate(sentiment_data)

models = [
    torch.load("./models/m2.6.4.64"),
    torch.load("./models/m1.6.5.32"),
    torch.load("./models/m2.7.5.32"),
    torch.load("./models/m2.4.5.32"),
]
for model in models:
    text = """
The stock might do good but im not sure. it might turn out bad
    """
    prediction = model.predict(processed_sentiment, text)
    print("{:<30} | {:<}".format(text[1:-1][:30], raw_data.labels[prediction[0]]))