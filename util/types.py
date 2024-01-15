from enum import Enum
BatchType = Enum('BatchType', ['train', 'valid', 'test'])
TrainPhase = Enum('TrainPhase', ['base_init', 'base_weightgen'])
