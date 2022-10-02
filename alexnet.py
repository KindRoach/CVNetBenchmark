import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights
from torchvision.transforms import transforms


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def image_transforms(batch):
        batch["image"] = [preprocess(image.convert("RGB")) for image in batch["image"]]
        return batch

    dataset.set_transform(image_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def create_test_batch(batch_size: int) -> torch.Tensor:
    return torch.rand([batch_size, 3, 244, 244], dtype=torch.float)


def load_pretrained_model():
    return torch.hub.load("pytorch/vision:v0.13.1", "alexnet", weights=AlexNet_Weights.DEFAULT)
