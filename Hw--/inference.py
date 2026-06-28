import torch
import torchvision.transforms as T
from PIL import Image
from data.dataset import FlickrDataset
from models.captioner import ImageCaptioner
from utils.visualization import Visualizer


class CaptionGenerator:
    def __init__(self, model_path, dataset, embed_size=512, num_heads=8, num_layers=4, forward_expansion=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        self.model = ImageCaptioner(
            embed_size, len(dataset.vocab), num_heads, num_layers, forward_expansion, dropout=0.0
        ).to(self.device)

        # Загрузка предобученных весов
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def predict(self, image_path, max_length=20):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        caption_indices = [self.dataset.vocab.stoi["<SOS>"]]

        with torch.no_grad():
            # Получаем Memory из CNN один раз
            memory = self.model.encoder(img_tensor)

            for _ in range(max_length):
                tgt_tensor = torch.tensor(caption_indices).unsqueeze(0).to(self.device)
                predictions = self.model.decoder(tgt_tensor, memory)

                next_word_idx = predictions[0, -1, :].argmax().item()
                caption_indices.append(next_word_idx)

                if next_word_idx == self.dataset.vocab.stoi["<EOS>"]:
                    break

        result = [self.dataset.vocab.itos[idx] for idx in caption_indices]
        return " ".join(result[1:-1])


# Пример вызова генерации
if __name__ == "__main__":
    # Фиктивная инициализация датасета для доступа к Vocabulary
    transform_stub = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = FlickrDataset(root_dir="data/Images", captions_file="data/captions.txt", transform=transform_stub)

    generator = CaptionGenerator(model_path="resnet_transformer_captioner.pth", dataset=ds)

    test_img = "data/Images/sample.jpg"
    predicted_text = generator.predict(test_img)

    # Отображаем картинку с предсказанием
    Visualizer.show_prediction(test_img, predicted_text)
