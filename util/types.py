from enum import Enum
BatchType = Enum('BatchType', ['train', 'valid', 'test'])
TrainPhase = Enum('TrainPhase', ['base_init', 'base_test', 'base_weightgen', 'novel_val', 'novel_test'])
DatasetType = Enum('DatasetType', ['base', 'novel'])
ModelName = Enum('ModelName', ['samplecnn', 'cnn14'])
DatasetName = Enum('DatasetName', ['esc50'])
