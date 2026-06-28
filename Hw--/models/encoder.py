import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        # Загружаем предобученный ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)  # Замораживаем базовые слои CNN

        # Извлекаем карту признаков без слоев адаптивного пулинга и классификации
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Слой проекции: 2048 каналов ResNet переводим в размерность эмбеддинга трансформера
        self.linear = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images)  # Выход: [Batch, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1)  # Пересобираем: [Batch, 7, 7, 2048]
        features = features.view(features.size(0), -1,
                                 features.size(3))  # Выпрямляем пространственную сетку: [Batch, 49, 2048]
        return self.relu(self.linear(features))  # Результат: [Batch, 49, embed_size]
