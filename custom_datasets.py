import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
import os
import subprocess
import zipfile

import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from glob import glob



def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def mnist_dataset():
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    return dataset

def fashion_mnist_dataset():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    return dataset



def pets_dataset():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalisation des images RGB
    ])

    dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform, target_types="category")

    return dataset

def pets_dataset():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])
    
    #if you have private cats data please put it in a folder and place it in "./data/cats_data"; otherwise we use OxfordIIITPet dataset
    try :
        dataset = datasets.ImageFolder(root="./data/cats_data", transform=transform)
    except :
        dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform, target_types="category")

    return dataset


#faces dataset : data/thumbnails128x128.zip
def faces_dataset():
    file_path = 'data/thumbnails128x128.zip'
    
    if not os.path.exists('./data/faces/thumbnails128x128'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('./data')

    
    dataset_dir = './data/faces'
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    return dataset




def speechcommands_dataset(n=100, sample_rate=8000, audio_length=8000, n_mels=32):
    dataset_path = "./data/SpeechCommands"
    if not os.path.exists(dataset_path):
        print("Téléchargement du dataset...")
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)
    else:
        print("Dataset déjà présent. Chargement en cours...")
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=False)

    digit_labels = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    
    digit_samples = [(waveform, sample_rate) for waveform, sample_rate, label, _, _ in dataset if label in digit_labels]

    num_samples = min(n, len(dataset))
    dataset = torch.utils.data.Subset(digit_samples, list(range(num_samples)))

    def preprocess(batch, target_sample_rate, n_mels, audio_length):
        waveform, sample_rate = batch
    
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 


        resample = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)

        waveform = torch.nn.functional.pad(waveform, (0, max(0, audio_length - waveform.shape[-1])))

        mel_spectrogram = MelSpectrogram(sample_rate=target_sample_rate, n_mels=n_mels)(waveform)

        mel_spectrogram = mel_spectrogram[:, :, :audio_length] 

        return mel_spectrogram

    dataset = [preprocess(digit_samples[i], sample_rate, n_mels, audio_length) for i in range(len(digit_samples))]
    torch.cuda.empty_cache()
    return dataset






def get_dataset(name, n=10000):
    """
    Get the specified dataset.
    Args:
        name (str): Name of the dataset ("moons", "dino", "line", "circle", or "mnist").
        n (int, optional): Number of samples to use. Defaults to 8000.
    Returns:
        TensorDataset: The requested dataset.
    """
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "mnist":
        return mnist_dataset()
    elif name == "fashion-mnist":
        return fashion_mnist_dataset()
    elif name == "pets":
        return pets_dataset()
    elif name == "faces":
        return faces_dataset()
    elif name == "speech-commands":
        return speechcommands_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")