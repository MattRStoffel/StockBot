import torch
from TrainingLogger import TrainingLogger

# Need better class name 
class ModelTrainingManager:
    def __init__(self, model, learning_rate : int, data_count = 1000) -> None:
        #Initiate an instance
        self.total_accu = None
        self.model = model
        self.model.initial_LR = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()                                        # Defining loss function (used to measure the difference between the predicted output and the target) 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)    # Stochastic Gradient Descent optimizer used in the scheduler for finding local minimums
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)    # reduces the efect of the optimizer lowering learning rate at specified times (gamma)
        self.logger = TrainingLogger(data_count)

    def train_one_epoch(self, epoch, dataloader):
        self.model.train() #Set model to training mode
        
        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)

            loss = self.criterion(predicted_label, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            self.logger.log_train_progress(predicted_label, label, idx, epoch, dataloader)
        self.model.trained_epochs += 1

    def evaluate_accuracy(self, dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for label, text, offsets in dataloader:
                predicted_label = self.model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def train_and_evaluate(self, epochs, train_dataloader, valid_dataloader):
        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch, train_dataloader)
            accu_val = self.evaluate_accuracy(valid_dataloader)
            if self.total_accu is not None and self.total_accu > accu_val:
                self.scheduler.step()
            else:
                self.total_accu = accu_val
            self.logger.log_epoch_summary(epoch, accu_val)