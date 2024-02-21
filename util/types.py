from enum import Enum
BatchType = Enum('BatchType', ['train', 'valid', 'test'])
TrainPhase = Enum('TrainPhase', ['base_init', 'base_weightgen', 'novel_val', 'novel_test'])
DatasetType = Enum('DatasetType', ['base', 'novel'])
