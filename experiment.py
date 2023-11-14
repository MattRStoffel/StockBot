# import torch
from torch import tensor, load, no_grad, save
from TextDataLoader import TextDataLoader
from SentimentDataSource import SentimentDataSource 
from ModelTrainingManager import ModelTrainingManager
from TextPreprocessor import TextPreprocessor
from TextClassificationModel import TextClassificationModel

EPOCHS = 9
LR = 5
BATCH_SIZE = 64

processed_text = TextPreprocessor(SentimentDataSource())
data = TextDataLoader(SentimentDataSource(), processed_text, BATCH_SIZE)

def make_model(EPOCHS, LR, embedding_size):
    model = TextClassificationModel(len(processed_text.vocab), embedding_size, data.number_of_classes).to("cpu")

    trainer = ModelTrainingManager(model, LR, len(data.train))

    for E in range(1,EPOCHS):
        trainer.train_and_evaluate(1, data.train, data.valid)
        model = trainer.model
        if E > 6:
            save(model, "model." + str(E) + "_" + str(LR) + "_" + str(embedding_size))

def predict(model, text):
    with no_grad(): # disables gradient calculation
        text = tensor(processed_text.text_pipeline(text))
        output = model(text, tensor([0]))
        return (output.argmax(1).item() + 1, output.tolist())
    
def thredded_train():
    from concurrent.futures import ThreadPoolExecutor

    def make_model_wrapper(args):
        EP, L, em = args
        make_model(EP, L, em)

    # Define the parameter ranges
    L_range = [3,4]
    em_range = [16, 32]

    # Generate all combinations of parameters
    parameter_combinations = [(20, L, em) for L in L_range for em in em_range]

    # Specify the number of threads you want to use
    num_threads = 12  # Adjust as needed

    # Use ThreadPoolExecutor to parallelize the loop
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(make_model_wrapper, parameter_combinations)


def visual_test():
    bullish_tweets_hedge = [
        "Just heard some exciting news about the market! Looks like hedge funds are going all in. #BullishMarket 🚀",
        "Feeling optimistic about the future! Hedge funds seem confident in the market's upward trend. #BullishVibes 💹",
        "Bulls are running wild today! Hedge funds are placing big bets on the market's success. #BullishRun 🐂",
        "Market sentiment is on fire! Hedge funds are showing strong support for the bullish trend. #BullishOutlook 🔥",
        "Hedge funds are doubling down on their positions. Looks like a bullish ride ahead! #MarketOptimism 📈",
        "Seeing a lot of positive signals from hedge funds lately. Bull market enthusiasts, rejoice! #BullishSignals 📊",
        "Exciting times in the market! Hedge funds are expressing confidence in the ongoing bullish momentum. #BullishConfidence 🌐",
        "Bullish sentiment is spreading like wildfire, especially among hedge funds. Market outlook is looking bright! #BullishMarket 💡",
        "Hedge funds are placing their bets on a strong market rally. Looks like a bullish wave is coming! #BullishWave 🌊",
        "Market bulls are in control! Hedge funds are contributing to the positive atmosphere. #BullishMarketLeaders 🐃"
    ]

    bullish_tweets_banks = [
        "Just heard some exciting news about the market! Looks like banks are going all in. #BullishMarket 🚀",
        "Feeling optimistic about the future! Banks seem confident in the market's upward trend. #BullishVibes 💹",
        "Bulls are running wild today! Banks are placing big bets on the market's success. #BullishRun 🐂",
        "Market sentiment is on fire! Banks are showing strong support for the bullish trend. #BullishOutlook 🔥",
        "Banks are doubling down on their positions. Looks like a bullish ride ahead! #MarketOptimism 📈",
        "Seeing a lot of positive signals from banks lately. Bull market enthusiasts, rejoice! #BullishSignals 📊",
        "Exciting times in the market! Banks are expressing confidence in the ongoing bullish momentum. #BullishConfidence 🌐",
        "Bullish sentiment is spreading like wildfire, especially among banks. Market outlook is looking bright! #BullishMarket 💡",
        "Banks are placing their bets on a strong market rally. Looks like a bullish wave is coming! #BullishWave 🌊",
        "Market bulls are in control! Banks are contributing to the positive atmosphere. #BullishMarketLeaders 🐃"
    ]

    bearish_tweets_hedge = [
        "Concerned about the market lately. Hearing that hedge funds are pulling back. #BearishMarket 📉",
        "Seeing signs of caution in the market. Hedge funds seem to be adopting a bearish stance. #BearishOutlook 🚨",
        "Market bears gaining momentum as hedge funds reduce exposure. #BearishTrend 🐻",
        "Hedge funds appear to be hedging their bets. Market sentiment turning bearish. #MarketCaution 📉",
        "Noticing a shift in market dynamics. Hedge funds are taking a more conservative approach. #BearishSignals 🚩",
        "Word on the street is hedge funds are getting skeptical. Brace yourselves for a bearish phase. #MarketConcerns 📉",
        "Market pessimism creeping in as hedge funds express doubts. #BearishSentiment 🐻",
        "Hedge funds seem to be stepping back, signaling a potential downturn. #BearishMarketAlert 🚨",
        "Bearish clouds gathering as hedge funds adjust their positions. #MarketWorries 🌧️",
        "Hedge funds taking defensive measures. Market outlook appears bearish. #BearishTrends 📉"
    ]

    bearish_tweets_banks = [
        "Concerned about the market lately. Hearing that banks are pulling back. #BearishMarket 📉",
        "Seeing signs of caution in the market. Banks seem to be adopting a bearish stance. #BearishOutlook 🚨",
        "Market bears gaining momentum as banks reduce exposure. #BearishTrend 🐻",
        "Banks appear to be hedging their bets. Market sentiment turning bearish. #MarketCaution 📉",
        "Noticing a shift in market dynamics. Banks are taking a more conservative approach. #BearishSignals 🚩",
        "Word on the street is banks are getting skeptical. Brace yourselves for a bearish phase. #MarketConcerns 📉",
        "Market pessimism creeping in as banks express doubts. #BearishSentiment 🐻",
        "Banks seem to be stepping back, signaling a potential downturn. #BearishMarketAlert 🚨",
        "Bearish clouds gathering as banks adjust their positions. #MarketWorries 🌧️",
        "Banks taking defensive measures. Market outlook appears bearish. #BearishTrends 📉"
    ]

    test_tests = [bullish_tweets_hedge, bullish_tweets_banks, bearish_tweets_hedge, bearish_tweets_banks]
    test_tests_names = ["bullish_tweets_hedge", "bullish_tweets_banks", "bearish_tweets_hedge", "bearish_tweets_banks"]

    model = load("1model.13_4_16")

    labels = SentimentDataSource().labels
    for i, s in enumerate(test_tests):
        print("-" * 80)
        print(test_tests_names[i])
        print("-" * 80)
        for test in s:
            p = predict(model, test)
            print("| {:<40} | {:<20} | {}".format(test[:40],labels[p[0]],p[1]))

def batch_test():
    # Define the parameter ranges
    E_range = range(7, 20)
    L_range = [3,4]
    em_range = [16, 32]

    # Generate all combinations of parameters
    parameter_combinations = [(E, L, em)for E in E_range for L in L_range for em in em_range]
    models = []
    for (E, L, em) in parameter_combinations:
        name = "model." + str(E) + "_" + str(L) + "_" + str(em)
        model = load(name)
        trainer = ModelTrainingManager(model, LR, len(data.train)) 
        accuracy = trainer.evaluate_accuracy(data.test)
        models.append((accuracy, name))

    models = sorted(models)
    for model in models[:4]:
        print("{:13} | {:3f}".format(model[1], model[0]))
        

visual_test()