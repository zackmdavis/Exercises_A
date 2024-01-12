import pathlib
import time

import torch
import numpy as np

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm



class Autoencoder(nn.Module):
    def __init__(self, image_size, latent_dimensionality, hidden_dimensionality):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (image_size//4)**2, hidden_dimensionality),
            nn.ReLU(),
            nn.Linear(hidden_dimensionality, latent_dimensionality),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimensionality, hidden_dimensionality),
            nn.ReLU(),
            nn.Linear(hidden_dimensionality, 32 * (image_size//4)**2),
            nn.Unflatten(1, (32, image_size//4, image_size//4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderTrainer:
    def __init__(
            self,
            latent_dimensionality=5,
            hidden_dimensionality=128,
            batch_size=32,
            epochs=3,
            lr=0.001,
            betas=(0.5, 0.999),
            seconds_between_evals=5
    ):

        self.image_size = 28
        self.latent_dimensionality = latent_dimensionality
        self.hidden_dimensionality = hidden_dimensionality
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.betas = betas
        self.seconds_between_evals = seconds_between_evals

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(
            root=pathlib.Path.cwd() / "data",
            transform=transform,
            download=True,
        )
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = nn.MSELoss()

        self.model = Autoencoder(
            self.image_size, self.latent_dimensionality, self.hidden_dimensionality
        ).to(self.device).train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)

        self.step = 0

    def training_step(self, input_):
        output = self.model(input_)
        loss = self.criterion(input_, output)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @torch.inference_mode()
    def evaluate(self, input_):
        self.model.eval()
        output = self.model(input_)

        samples = (output - output.min()) / (output.max() - output.min())
        samples = (samples * 255).byte()
        image_array = np.transpose(samples.cpu().numpy(), (0, 2, 3, 1))
        image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], image_array.shape[2])
        # Save each image in the batch
        for i in range(4):
            img = Image.fromarray(image_array[i])
            img.save("step{}_{}.png".format(self.step, i))

        print("saved evaluation images for step {}!".format(self.step))

        self.model.train()

    def train(self):
        last_eval_time = time.time()
        for epoch in range(self.epochs):
            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            for batch, _ in progress_bar:
                batch = batch.to(self.device)
                # at eval time, do it before training on those examples
                if time.time() - last_eval_time > self.seconds_between_evals:
                    last_eval_time = time.time()
                    self.evaluate(batch)

                loss = self.training_step(batch)
                self.step += batch.size(0)
                progress_bar.set_description("step={} loss={}".format(self.step, loss))


if __name__ == "__main__":
    AutoencoderTrainer().train()

# Looks pretty good!
