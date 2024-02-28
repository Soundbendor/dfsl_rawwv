from enum import Enum
BatchType = Enum('BatchType', ['train', 'valid', 'test'])
TrainPhase = Enum('TrainPhase', ['base_init', 'base_test', 'base_weightgen', 'base_trainall', 'novel_valid', 'novel_test', 'run_all'])
DatasetType = Enum('DatasetType', ['base', 'novel'])
ModelName = Enum('ModelName', ['samplecnn', 'cnn14'])
DatasetName = Enum('DatasetName', ['esc50'])
