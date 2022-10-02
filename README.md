# Classic Computer Vision Network Benchmark

This is repo for classic computer vision network benchmark
on the image classification task with dataset [imagenet-1k](https://www.image-net.org/).
There are two indicators: [top-5 accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html)
and performance in ```sample/s```.

## Benchmark Setup

### Hardware

- Network connection to [Hugging Face](https://huggingface.co/)
- Free disk space at least 320GB.
- [Optional] Pytorch hardware acceleration device.

### Python Environment

[Conda](https://docs.conda.io/en/latest/) as example:

```bash
conda create --name cvmark python=3.9
conda activate cvmark
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c huggingface -c conda-forge datasets
conda install -c conda-forge scikit-learn
```

### Huggingface Access Tokens

Downloading [imagenet-1k](https://www.image-net.org/) from Huggingface
requires you to agree to the [User Agreement](https://huggingface.co/datasets/imagenet-1k). So you have to register
a [Huggingface Account](https://huggingface.co/join) and create a [Huggingface Access Token](https://huggingface.co/docs/hub/security-tokens).
Once you gain token, put it in a python script ```config.py```:

```python
# See: https://huggingface.co/docs/hub/security-tokens
# Put your huggingface Access Tokens here
hf_token = "hf_**********************************"
```

so that ```main.py``` could import ```hf_token```.

## Run Benchmark

The first run will take time to download the dataset and pretrained model.
The good news is that they're all cached on disk.

```bash
python main.py
```
