import time


class TrainingLogger:
    def __init__(self, data_count: int = 1000, log_intervals : int = 3, output = lambda x: print(x)) -> None:
        self.total_acc, self.total_count = 0, 0
        self.log_interval = int(data_count/(log_intervals+1) + (data_count/(log_intervals * 20)))
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.output = output

    def log_train_progress(self, predicted_label, label, idx, epoch, dataloader):
        self.total_acc += (predicted_label.argmax(1) == label).sum().item()
        self.total_count += label.size(0)
        if self.log_interval != 0 and idx % self.log_interval == 0 and idx > 0:
            self.output(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), self.total_acc / self.total_count
                )
            )
            self.total_acc, self.total_count = 0, 0
            self.start_time = time.time()

    def log_epoch_summary(self, epoch, accu_val):
        #Log progress
        self.output("-" * 59)
        self.output(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - self.epoch_start_time, accu_val
            )
        )
        self.output("-" * 59)
        