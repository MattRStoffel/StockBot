import torch
import time

from TextClassificationModel import TextClassificationModel

def train(dataloader):
    #for logging purposes
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    model.train() #Set model to training mode

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)

        loss = criterion(predicted_label, label)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        #Log progress
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            # loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


#Initiate an instance
total_accu = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Data import vocab_size, embedding_size, number_of_classes
model = TextClassificationModel(vocab_size, embedding_size, number_of_classes).to(device)

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate

# Defining loss function 
# (used to measure the difference between the predicted output and the target)
criterion = torch.nn.CrossEntropyLoss() 

# Stochastic Gradient Descent optimizer 
# used in the scheduler for finding local minimums
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# reduces the efect of the optimizer (lowering learning rate) at specified times (gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

from Data import train_dataloader, valid_dataloader, test_dataloader
# run the model
for epoch in range(1, EPOCHS + 1):
    #for loging progress
    epoch_start_time = time.time()

    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    
    #Log progress
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

#Evaluate the model with test dataset
print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

torch.save(model, "model.h1")
