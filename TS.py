import argparse
import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import custom_datasets as datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Noise Scheduler for Time Series   

seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.1):
    betas = torch.linspace(beta_start, beta_end, num_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

class NoiseSchedulerTS:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.1, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        self.betas, self.alphas, self.alpha_bars = make_beta_schedule(num_steps, beta_start, beta_end)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
    
    def add_noise(self, x_start, noise, t):
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().view(-1, 1)
        sqrt_one_minus = (1 - self.alpha_bars[t]).sqrt().view(-1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def step(self, model_output, t, sample):
        t = int(t)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        x0_pred = (sample - (1 - alpha_bar_t).sqrt() * model_output) / alpha_t.sqrt()
        mean = x0_pred
        noise = torch.randn_like(sample) if t > 0 else 0
        return mean + beta_t.sqrt() * noise



# TimeGrad Model                    

# class SimpleRNN(nn.Module): (no dropout, unidirectional)
#     def __init__(self, input_dim=1, hidden_dim=32):
#         super(SimpleRNN, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
#     def forward(self, x, h0=None):
#         out, h = self.gru(x, h0)
#         return out, h
    
#     def init_hidden(self, batch_size):
#         return torch.zeros(1, batch_size, self.gru.hidden_size)


class SimpleRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, dropout=0.0, bidirectional=False):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # If more than one layer, we apply dropout between layers
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0, 
            batch_first=True, 
            bidirectional=bidirectional
        )
    
    def forward(self, x, h0=None):
        # Automatically initialize hidden state if not provided.
        if h0 is None:
            h0 = self.init_hidden(x.size(0))
        out, h = self.gru(x, h0)
        return out, h
    
    def init_hidden(self, batch_size):
        num_directions = 2 if self.bidirectional else 1
        # Shape: (num_layers * num_directions, batch_size, hidden_dim)
        device = next(self.parameters()).device  # Ensures hidden state is on the same device as model parameters
        return torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)



class Denoiser(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, embed_dim=32, num_steps=100):
        super(Denoiser, self).__init__()
        self.embedding = nn.Embedding(num_steps, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x_noisy, h, t):
        h = h.squeeze(0)
        t_emb = self.embedding(t)
        inp = torch.cat([x_noisy, h, t_emb], dim=1)
        return self.fc(inp)

class TimeGradModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_steps=100, num_layers=2, dropout=0.2, bidirectional=True):
        super(TimeGradModel, self).__init__()
        self.rnn = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.denoiser = Denoiser(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=32, num_steps=num_steps)
        self.num_steps = num_steps
    
    def forward(self, x_noisy, h, t):
        return self.denoiser(x_noisy, h, t)


# Training Loop                     

def train_timegrad(model, noise_scheduler, data, seq_len=40, num_epochs=1000, lr=1e-4, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = data.shape[0]
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1).to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0
        count = 0
        for start in range(T - seq_len):
            subseq = data[start:start+seq_len]
            subseq = subseq.unsqueeze(0)
            out, h_tmp = model.rnn(subseq[:, :-1, :])
            n = torch.randint(low=0, high=noise_scheduler.num_steps, size=(1,), device=device)
            x_clean = subseq[:, -1, :]
            noise = torch.randn_like(x_clean)
            x_noisy = noise_scheduler.add_noise(x_clean, noise, n)
            noise_pred = model(x_noisy, h_tmp, n)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
        
        if (epoch+1) % (num_epochs/10) == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss / count:.4f}")

@torch.no_grad()
def forecast_timegrad(model, noise_scheduler, context, forecast_steps=10, device='cpu'):
    model.eval()
    context = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) #.unsqueeze(-1) for electricity
    batch_size = 1
    h = model.rnn.init_hidden(batch_size).to(device)
    _, h = model.rnn(context, h)
    
    forecast = []
    for step in range(forecast_steps):
        x = torch.randn(batch_size, 1, device=device)
        for t in reversed(range(noise_scheduler.num_steps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            eps_pred = model(x, h, t_tensor)
            x = noise_scheduler.step(eps_pred, t, x)
        forecast.append(x.item())
        x_input = x.unsqueeze(0)
        _, h = model.rnn(x_input, h)
    return forecast


# Main Execution Function          

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset using provided dataset name and number of points
    ds = datasets.get_dataset(args.dataset, n=args.n_data)
    data_tensor = ds[:][0]
    data_array = data_tensor.squeeze(-1).numpy()

    # Set hyperparameters from command-line arguments
    num_steps = args.num_steps
    lr = args.lr
    num_epochs = args.num_epochs
    context_length = args.context_length
    forecast_steps = args.forecast_steps
    
    # Initialize diffusion components for the TimeGrad model
    noise_scheduler = NoiseSchedulerTS(num_steps=num_steps, device=device)
    model = TimeGradModel(input_dim=1, hidden_dim=args.hidden_dim, num_steps=num_steps,     num_layers=args.num_layers,
    dropout=args.dropout,
    bidirectional=args.bidirectional).to(device)
    
    # Prepare the context from the dataset
    context_series = data_array[:context_length]
    
    # Train the diffusion model
    print(f"Training with lr={lr}, num_steps={num_steps}, context_length={context_length}, "
          f"num_epochs={num_epochs}, hidden_dim={args.hidden_dim} on dataset: {args.dataset}")
    train_timegrad(model, noise_scheduler, data_array, num_epochs=num_epochs, lr=lr, device=device)
    
    # Forecast future steps
    true_future = data_array[context_length-1:context_length + forecast_steps]  # to have a continuous plot
    forecast = forecast_timegrad(model, noise_scheduler, context_series, forecast_steps=forecast_steps, device=device)
    
    # Calculate and print error metrics (MSE and MAE)
    mse = mean_squared_error(true_future[1:], forecast)
    mae = mean_absolute_error(true_future[1:], forecast)
    print(f"Forecast MSE: {mse:.4f}")
    print(f"Forecast MAE: {mae:.4f}")
    
    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(context_length), context_series, label="Context")
    plt.plot(np.arange(context_length-1, context_length + forecast_steps), true_future, 
             label="True Future", color="green")
    plt.plot(np.arange(context_length, context_length + forecast_steps), forecast, 
             label="Forecast", marker ="o", color="orange")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Forecast using TimeGrad DDPM on {args.dataset} Dataset")
    
    # Depending on the flag, either display or save the plot
    if args.save_plot:
        outdir = args.outdir if args.outdir is not None else "./output"
        os.makedirs(outdir, exist_ok=True)
        # Create filename using the tested parameter name and value if provided
        if args.test_param and args.test_value:
            file_name = f"{args.test_param}_{args.test_value}_forecast_plot.png"
            array_name = f"{args.test_param}_{args.test_value}_forecast.npy"
        else:
            file_name = "forecast_plot_optim.png"
            array_name = "forecast_optim.npy"
        plot_path = os.path.join(outdir, file_name)
        plt.savefig(plot_path)
        array_path = os.path.join(outdir, array_name)
        np.save(array_path, forecast)
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()


# Entry Point with Argument Parsing 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TimeGrad DDPM forecasting with adjustable hyperparameters."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--context_length", type=int, default=100, help="Context length for forecasting")
    parser.add_argument("--forecast_steps", type=int, default=20, help="Number of forecast steps")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for the TimeGrad model")
    
    # Dataset-specific arguments
    parser.add_argument("--dataset", type=str, default="electricity", help="Name of dataset to test (e.g., 'electricity')")
    parser.add_argument("--n_data", type=int, default=10000, help="Number of data points to load from the dataset")
    
    # Option to save the plot instead of showing it
    parser.add_argument("--save_plot", action="store_true", help="Save the plot as an image instead of displaying it")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory to save the plot (default: ./output)")
    
    # New arguments for naming the output file based on the parameter tested
    parser.add_argument("--test_param", type=str, default="", help="Name of the hyperparameter tested (e.g., 'lr')")
    parser.add_argument("--test_value", type=str, default="", help="Value of the hyperparameter tested (e.g., '0.001')")

    parser.add_argument("--num_layers", type=int, default=1, help="Number of GRU layers in the RNN")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate between RNN layers")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional GRU")

    args, unknown = parser.parse_known_args()
    main(args)
