import time
from dataclasses import dataclass
from typing import Dict

import pandas
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import *
from torchvision.models._api import WeightsEnum
from torchvision.transforms import transforms
from tqdm import tqdm

from config import hf_token

REPO_URL = "pytorch/vision:v0.13.1"


@dataclass
class ModelConfig:
    weight: WeightsEnum
    resize: int
    crop: int


# See more models in https://pytorch.org/vision/stable/models
MODEL_CONFIG: Dict[str, ModelConfig] = {
    alexnet.__name__: ModelConfig(AlexNet_Weights.DEFAULT, 256, 224),
    vgg19_bn.__name__: ModelConfig(VGG19_BN_Weights.DEFAULT, 256, 224),
    googlenet.__name__: ModelConfig(GoogLeNet_Weights.DEFAULT, 256, 224),
    inception_v3.__name__: ModelConfig(Inception_V3_Weights.DEFAULT, 342, 299),
    resnet50.__name__: ModelConfig(ResNet50_Weights.DEFAULT, 232, 224),
    mobilenet_v3_large.__name__: ModelConfig(MobileNet_V3_Large_Weights.DEFAULT, 256, 224),
}


def acc_benchmark(model_name: str, batch_size: int, device: str) -> float:
    torch.hub.set_dir("model")
    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir="data",
        use_auth_token=hf_token,
        ignore_verifications=False,
    )
    labels = dataset["label"]

    dataloader = create_dataloader(model_name, dataset, batch_size)
    model = torch.hub.load(REPO_URL, model_name, weights=MODEL_CONFIG[model_name].weight)

    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    predicts = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Model inference", unit="batch"):
            output = model(batch["image"].to(torch_device))
            softmax = torch.nn.functional.softmax(output, dim=1)
            predicts.append(softmax)
    predicts = torch.cat(predicts).cpu().numpy()
    return top_k_accuracy_score(labels, predicts, k=5)


def create_dataloader(model_name: str, dataset: Dataset, batch_size: int) -> DataLoader:
    config = MODEL_CONFIG[model_name]
    preprocess = transforms.Compose([
        transforms.Resize(config.resize),
        transforms.CenterCrop(config.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def image_transforms(batch):
        batch["image"] = [preprocess(image.convert("RGB")) for image in batch["image"]]
        return batch

    dataset.set_transform(image_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def performance_benchmark(model_name: str, batch_size: int, device: str) -> float:
    torch_device = torch.device(device)
    config = MODEL_CONFIG[model_name]
    test_batch = torch.rand([batch_size, 3, config.crop, config.crop], dtype=torch.float).to(torch_device)

    model = torch.hub.load(REPO_URL, model_name, weights=MODEL_CONFIG[model_name].weight)
    model.eval()
    model.to(torch_device)

    with torch.no_grad(), tqdm(desc="Model inference", unit="batch") as pbar:
        timeout = 30  # seconds
        time_end = time.time() + timeout
        while time.time() < time_end:
            model(test_batch)
            pbar.update()

    token = pbar.__str__().split(",")[1].split("/")[0]
    if "batch" in token:  # e.g: 1.25batch
        samples_per_second = float(token.split("b")[0]) * batch_size
    else:  # e.g: 3.78s
        samples_per_second = 1 / float(token.split("s")[0])
    return samples_per_second


def main():
    result = {"model": [], "top-5 acc": [], "sample/s": []}
    for model_name in MODEL_CONFIG.keys():
        acc = acc_benchmark(model_name, 64, "cuda:0")
        performance = performance_benchmark(model_name, 64, "cuda:0")
        result["model"].append(model_name)
        result["top-5 acc"].append(acc)
        result["sample/s"].append(performance)
    print(pandas.DataFrame(result))


if __name__ == '__main__':
    main()
