import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid

import custom_datasets as datasets
from positional_embeddings import PositionalEmbedding
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb).to(device)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0).to(device)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0).to(device)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers).to(device)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0].to(device))
        x2_emb = self.input_mlp2(x[:, 1].to(device))
        t_emb = self.time_mlp(t.to(device))
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x
    
    
    
    
class ResBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, mid_ch=None, residual=False):
        super(ResBlock, self).__init__()
        
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.resnet_conv = nn.Sequential(
            nn.Conv2d(inp_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch)
        ).to(device)
        
    def forward(self, x):
        x = x.to(device)
        if self.residual:
            return x + self.resnet_conv(x)
        else:
            #print("shape x before resnet conv", x.shape)
            return self.resnet_conv(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()
        
        self.attn_norm = nn.GroupNorm(num_groups=8, num_channels=channels).to(device)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True).to(device)
        
    def forward(self, x):
        x = x.to(device)
        b, c, h, w = x.shape
        inp_attn = x.reshape(b, c, h*w)
        inp_attn = self.attn_norm(inp_attn)
        inp_attn = inp_attn.transpose(1, 2)
        out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
        out_attn = out_attn.transpose(1, 2).reshape(b, c, h, w)
        return x + out_attn
    
class DownBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256):
        super(DownBlock, self).__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch)
        ).to(device)
        
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch)
        ).to(device)
        
    def forward(self, x, t):
        x = x.to(device)
        x = self.down(x)

        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        t_emb = t_emb.to(device)
        return x + t_emb

class UpBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256):
        super(UpBlock, self).__init__()
        
        self.upsamp =  nn.UpsamplingBilinear2d(scale_factor=2).to(device)
        self.up = nn.Sequential(
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch, mid_ch=inp_ch//2)
        ).to(device)
        
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch)
        ).to(device)
        
    def forward(self, x, skip, t):
        x = x.to(device)
        x = self.upsamp(x)

        diff_h = skip.size()[2] - x.size()[2]
        diff_w = skip.size()[3] - x.size()[3]
        
        # padding to x if necessary
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.up(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        t_emb = t_emb.to(device)
        return x + t_emb
    
from positional_embeddings import PositionalEmbedding

class UNet(nn.Module):
    def __init__(self, t_emb_dim, in_channels, out_channels, emb_type="sinusoidal"):
        super(UNet, self).__init__()
        
        self.t_emb_dim = t_emb_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.time_embedding = PositionalEmbedding(size=t_emb_dim, type=emb_type)

        self.inp = ResBlock(inp_ch=in_channels, out_ch=64).to(device)
        self.down1 = DownBlock(inp_ch=64, out_ch=128, t_emb_dim=t_emb_dim).to(device)
        self.sa1 = SelfAttentionBlock(channels=128).to(device)
        self.down2 = DownBlock(inp_ch=128, out_ch=256, t_emb_dim=t_emb_dim).to(device)
        self.sa2 = SelfAttentionBlock(channels=256).to(device)
        self.down3 = DownBlock(inp_ch=256, out_ch=256, t_emb_dim=t_emb_dim).to(device)
        self.sa3 = SelfAttentionBlock(channels=256).to(device)
        
        self.lat1 = ResBlock(inp_ch=256, out_ch=512).to(device)
        self.lat2 = ResBlock(inp_ch=512, out_ch=512).to(device)
        self.lat3 = ResBlock(inp_ch=512, out_ch=256).to(device)
        
        self.up1 = UpBlock(inp_ch=512, out_ch=128, t_emb_dim=t_emb_dim).to(device)
        self.sa4 = SelfAttentionBlock(channels=128).to(device)
        self.up2 = UpBlock(inp_ch=256, out_ch=64, t_emb_dim=t_emb_dim).to(device)
        self.sa5 = SelfAttentionBlock(channels=64 ).to(device)
        self.up3 = UpBlock(inp_ch=128, out_ch=64, t_emb_dim=t_emb_dim).to(device)
        self.sa6 = SelfAttentionBlock(channels=64).to(device)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1).to(device)

    def forward(self, x, t):
        t = t.unsqueeze(1).float().to(device)  
        t_emb = self.time_embedding(t)  
        x1 = self.inp(x)
        x2 = self.down1(x1, t_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb)
        x4 = self.sa3(x4)
        
        x4 = self.lat1(x4)
        x4 = self.lat2(x4)
        x4 = self.lat3(x4)
        
        x = self.up1(x4, x3, t_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb)
        x = self.sa6(x)
        output = self.out(x)
        return output

    

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32).to(device) ** 2

        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.).to(device)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5).to(device)
        self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5).to(device)

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1).to(device)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)


    def reconstruct_x0(self, x_t, t, noise):
        x_t = x_t.to(device)
        noise = noise.to(device)
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        x_0 = x_0.to(device)
        x_t = x_t.to(device)
        
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        variance = variance.to(device)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(device)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(device)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        x_start = x_start.to(device)
        x_noise = x_noise.to(device)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
          
        if x_start.dim() == 4:  # this is an image (batch, channels, height, width)
            s1 = s1.view(-1, 1, 1, 1)  
            s2 = s2.view(-1, 1, 1, 1)
        elif x_start.dim() == 3:  # this is a Mel spectrogram (batch, n_mels, time_steps)
            s1 = s1.view(-1, 1, 1)  
            s2 = s2.view(-1, 1, 1)
        else :
            s1 = s1.reshape(-1, 1)
            s2 = s2.reshape(-1, 1)
        noisy = s1 * x_start + s2 * x_noise
        return noisy


    def __len__(self):
        return self.num_timesteps




##########################
from pydub import AudioSegment  
def save_audio(audio_tensor, filename, sample_rate, save_format="wav"):
    """Save audio tensor to disk in WAV or MP3 format."""
    audio_tensor = audio_tensor.squeeze(1) 
    if save_format == "wav":
        torchaudio.save(filename, audio_tensor, sample_rate)
    elif save_format == "mp3":
        
        audio_numpy = audio_tensor.cpu().numpy()
        audio_numpy = np.int16(audio_numpy * 32767)
        audio_segment = AudioSegment(
            audio_numpy.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit PCM
            channels=1
        )
        audio_segment.export(filename, format="mp3")
        
        ###################################
        
        
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons", "mnist", "fashion-mnist", "pets", "faces",  "speech-commands"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)

    config = parser.parse_args()

    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    # choose model based on dataset
    if config.dataset == "mnist" or config.dataset == "fashion-mnist" :
        model = UNet(t_emb_dim=config.embedding_size, in_channels=1, out_channels=1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
    elif config.dataset == "pets" or config.dataset == "faces":
        model = UNet(t_emb_dim=config.embedding_size, in_channels=3, out_channels=3)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
    elif config.dataset == "speech-commands":
        model = UNet(t_emb_dim=config.embedding_size, in_channels=1, out_channels=1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
    else:
        model = MLP(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding)
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
            )

        
    model.to(device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)


    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)
    np.random.seed(1111)
    print('GPU name:', torch.cuda.get_device_name(), '\n')
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            if config.dataset != "speech-commands":
                batch = batch[0]


            noise = torch.randn_like(batch).to(device)
            timesteps = torch.randint(low=0, high=noise_scheduler.num_timesteps, size=(batch.shape[0],)).long().to(device)
            batch = batch.to(device)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()
        if config.dataset != "speech-commands":
            if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
                model.eval()

                if config.dataset == "mnist" or config.dataset == "fashion-mnist" :
                    sample = torch.randn(config.eval_batch_size, 1, 32, 32).to(device)  # MNIST images
                elif config.dataset =="pets" or config.dataset == "faces":
                    sample = torch.randn(config.eval_batch_size, 3, 32, 32).to(device)
                elif config.dataset == "speech-commands":
                    sample = torch.randn(config.eval_batch_size, 1, 32, 41).to(device)

                else:
                    sample = torch.randn(config.eval_batch_size, 2).to(device)  # 2D data

                timesteps = list(range(len(noise_scheduler)))[::-1]
                for i, t in enumerate(tqdm(timesteps)):
                    t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                    with torch.no_grad():
                        residual = model(sample, t)
                    sample = noise_scheduler.step(residual, t[0], sample)

                frames.append(sample.cpu().numpy())


    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    
    #We wont save the audios during the training for memory issues
    # if config.dataset == "speech-commanords":
    #     print("Saving generated audio...")
    #     audio_dir = f"{outdir}/audio"
    #     os.makedirs(audio_dir, exist_ok=True)

    #     for i, sample in enumerate(frames):
    #         # Convert tensor to audio tensor and save it in the desired format
    #         audio_filename = f"{audio_dir}/{i:04}.{config.save_audio_format}"
    #         save_audio(torch.tensor(sample), audio_filename, 8000, "mp3")
    if config.dataset != "speech-commands":
        print("Saving images...")
        imgdir = f"{outdir}/images"
        os.makedirs(imgdir, exist_ok=True)
        frames = np.stack(frames)
        if config.dataset == "mnist" or config.dataset == "fashion-mnist" or  config.dataset == "pets" or config.dataset =="faces":
            for i, frame in enumerate(frames):
                frame = torch.from_numpy(frame)  
                grid = make_grid(frame, nrow=8, normalize=True, pad_value=1)  
                plt.figure(figsize=(10, 10))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
                plt.axis("off")
                plt.savefig(f"{imgdir}/{i:04}.png")
                plt.close()
        else:
            xmin, xmax = -6, 6
            ymin, ymax = -6, 6
            for i, frame in enumerate(frames):
                plt.figure(figsize=(10, 10))
                plt.scatter(frame[:, 0], frame[:, 1])
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.savefig(f"{imgdir}/{i:04}.png")
                plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)