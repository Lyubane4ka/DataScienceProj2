image_captioning/
│
├── data/
│   ├── dataset.py          # Классы Vocabulary и FlickrDataset
│   
│
├── models/
│   ├── encoder.py          # CNNEncoder на базе ResNet50
│   ├── decoder.py          # Transformer-декодер и Positional Encoding
│   └── captioner.py        # Общий класс-контейнер ImageCaptioner
│
├── utils/
│   └── visualization.py    # Функции отрисовки картинок и графиков Loss
│
├── Hw2.ipynb               # Скрипт запуска обучения
└── inference.py            # Скрипт генерации описания для новых картинок

