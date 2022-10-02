import time
from types import ModuleType

import pandas
import torch
from datasets import load_dataset
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

import alexnet
from config import hf_token


def acc_benchmark(m_model: ModuleType, batch_size: int, device: str) -> float:
    torch.hub.set_dir("model")
    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir="data",
        use_auth_token=hf_token,
        ignore_verifications=False,
    )
    labels = dataset["label"]

    dataloader = m_model.create_dataloader(dataset, batch_size)
    model = m_model.load_pretrained_model()

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


def performance_benchmark(m_model: ModuleType, batch_size: int, device: str) -> float:
    torch_device = torch.device(device)
    test_batch = m_model.create_test_batch(batch_size).to(torch_device)

    model = m_model.load_pretrained_model()
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
    for m_model in [alexnet]:
        acc = acc_benchmark(m_model, 128, "cuda:0")
        performance = performance_benchmark(m_model, 128, "cuda:0")
        result["model"].append(alexnet.__name__)
        result["top-5 acc"].append(acc)
        result["sample/s"].append(performance)
    print(pandas.DataFrame(result))


if __name__ == '__main__':
    main()
