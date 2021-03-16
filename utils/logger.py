import os


class AutomatedLogger():
    def __init__(self, savedir, enc=False):
        if (enc):
            automated_log_path = savedir + "/automated_log_encoder.txt"
        else:
            automated_log_path = savedir + "/automated_log.txt"
        if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
            with open(automated_log_path, "w") as myfile:
                myfile.write(f'epoch     loss-train     loss-val     IoU-train     IoU-val     learning-rate     time')

        self.automated_log_path = automated_log_path

    def write_raw(self, message: str):
        with open(self.automated_log_path, "a") as myfile:
            myfile.write(message)

    def write(self, epoch, loss_train, loss_val, iou_train, iou_val, lr, time):
        with open(self.automated_log_path, "a") as myfile:
            myfile.write(f'\n{epoch:<5}     {loss_train:10.4f}     {loss_val:8.4f}     {iou_train:9.4f}     {iou_val:7.4f}     {lr:13.8f}     {time:.4f}')
