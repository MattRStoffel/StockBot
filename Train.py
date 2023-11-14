import torch
from logger import logger

# Need better class name 
class trainer:
    def __init__(self, data, model, epochs : int, learning_rate : int) -> None:
        #Initiate an instance
        self.total_accu = None
        self.model = model
        self.train_data = data.train
        self.valid_data = data.valid
        self.epochs = epochs #epoch
        self.learning_rate = learning_rate  # learning rate
        self.criterion = torch.nn.CrossEntropyLoss()                                        # Defining loss function (used to measure the difference between the predicted output and the target) 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)    # Stochastic Gradient Descent optimizer used in the scheduler for finding local minimums
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)    # reduces the efect of the optimizer lowering learning rate at specified times (gamma)

    def train(self, epoch, dataloader):
        self.model.train() #Set model to training mode
        log = logger(len(dataloader), 6)

        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)

            loss = self.criterion(predicted_label, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            log.train(predicted_label, label, idx, epoch, dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for label, text, offsets in dataloader:
                predicted_label = self.model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def run(self):
        for epoch in range(1, self.epochs + 1):
            log = logger()
            self.train(epoch, self.train_data)
            accu_val = self.evaluate(self.valid_data)
            if self.total_accu is not None and self.total_accu > accu_val:
                self.scheduler.step()
            else:
                self.total_accu = accu_val
            log.epoch(epoch, accu_val)