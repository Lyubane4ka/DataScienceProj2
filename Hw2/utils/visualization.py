import matplotlib.pyplot as plt
from PIL import Image

class Visualizer:
    @staticmethod
    def plot_losses(train_losses, title="История обучения модели"):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
        plt.xlabel("Итерации (каждые 100 шагов)")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()

    @staticmethod
    def show_prediction(image_path, predicted_caption):
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Сгенерировано: \n{predicted_caption}", fontsize=12, color="darkgreen", weight="bold")
        plt.show()
