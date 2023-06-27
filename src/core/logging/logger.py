# import logging
# from sacred import Experiment
# from torch.utils.tensorboard import SummaryWriter
#
# __ALL__ = ['LoggerManager', 'TensorBoardLogger', 'PythonLogger', 'SacredLogger']
#
#
# class SacredLogger(Logger):
#     def __init__(self):
#         self.ex = Experiment('my_experiment')
#
#     def log(self, key, value):
#         self.ex.log_scalar(key, value)
#
#
# class TensorBoardLogger(Logger):
#     def __init__(self):
#         self.writer = SummaryWriter()
#
#     def log(self, key, value):
#         self.writer.add_scalar(key, value)
#
#
# class PythonLogger(Logger):
#     def __init__(self):
#         logging.basicConfig(filename='example.log', level=logging.INFO)
#
#     def log(self, key, value):
#         logging.info(f'{key}: {value}')
#
#
# class LoggerManager:
#     def __init__(self, *loggers):
#         self.loggers = loggers
#
#     def log(self, key, value):
#         for logger in self.loggers:
#             logger.log(key, value)
#
#     def log_config_params(self, config):
#         for k, v in config.items():
#             self.log(k, v)
#
# # usage example
# # def train_model(model, optimizer, data_loader, logger):
# #     for epoch in range(10):
# #         for batch in data_loader:
# #             # your training step here
# #             loss = model(batch)
#
# #             # Log the loss
# #             logger.log(f'loss_epoch_{epoch}', loss.item())
#
# #             # Log optimizer's learning rate
# #             logger.log(f'lr_epoch_{epoch}', optimizer.param_groups[0]['lr'])
#
# # # create a logger manager with TensorBoard and Python logging
# # logger = LoggerManager(TensorBoardLogger(), PythonLogger())
#
# # train your model
# # train_model(model, optimizer, data_loader, logger)
