from django.db import models
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# Create your models here.
# @title ExternalSongGenerationDataset

# all fields cannot be blank
class ExternalSongGenerationDataset(Dataset):
  def __init__(self, song_pianoroll, seq_length = 32):

    # Don't normalize anymore since it was done earlier
    self.data = song_pianoroll
    self.seq_length = seq_length
    self.length = int(song_pianoroll.size(1) / seq_length)

  def __getitem__(self, index):

    piano_sequence = self.data[0, (index*32):((index+1)*32), :]
    guitar_sequence = self.data[1, (index*32):((index+1)*32), :]
    bass_sequence = self.data[2, (index*32):((index+1)*32), :]
    strings_sequence = self.data[3, (index*32):((index+1)*32), :]
    drums_sequence = self.data[4, (index*32):((index+1)*32), :]

    return piano_sequence, guitar_sequence, bass_sequence, strings_sequence, drums_sequence

  def __len__(self):
    return self.length

# @title Conditional and Melody NN Architectures

# Conditional NN - uses current melody and previous harmony's LATENT vectors to predict next harmony's LATENT vectors
class ConditionalNN(nn.Module):
    def __init__(self, K):
        super(ConditionalNN, self).__init__()

        self.fc1 = nn.Linear(2*K, 128)
        self.fc2 = nn.Linear(128, K)

    def forward(self, prev_harmony, melody):

      x = torch.cat((prev_harmony, melody), axis = 1)
      x = F.relu(self.fc1(x))
      out = self.fc2(x)
      return out

# Melody NN - uses previous melody's LATENT vectors to predict next melody's LATENT VECTORS
class MelodyNN(nn.Module):
    def __init__(self, K):
        super(MelodyNN, self).__init__()
        self.fc1 = nn.Linear(K, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, K)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = F.relu(self.fc2(x))
      out = self.fc3(x)
      return out

# @title ConvVAE Architecture

class ConvVAE(nn.Module):
    def __init__(self, K, num_filters=32, filter_size=5):
        super(ConvVAE, self).__init__()

        # Define the recognition model (encoder or q) part
        # Input size: num_channels (1) x seq_length (32) x n_pitches (128)
        self.q_conv_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (4, 4), stride = (4, 4))
        self.q_conv_2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (4, 4), stride = (4, 4))
        self.q_conv_3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (2, 8), stride = (2, 8))
        self.q_fc_phi = nn.Linear(256, K+1)

        # Define the generative model (decoder or p) part
        self.p_fc_upsample = nn.Linear(K, 256)
        self.p_deconv_1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (2, 8), stride = (2, 8))
        self.p_deconv_2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (4, 4), stride = (4, 4))
        self.p_deconv_3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 1, kernel_size = (4, 4), stride = (4, 4))

        # Define a special extra parameter to learn scalar sig_x for all pixels
        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def infer(self, x):
        """Map (batch of) x to (batch of) phi which can then be passed to
        rsample to get z
        """
        x = x.unsqueeze(1)
        s = F.relu(self.q_conv_1(x))
        s = F.relu(self.q_conv_2(s))
        s = F.relu(self.q_conv_3(s))
        # Flatten s
        flat_s = s.view(s.size()[0], -1)
        phi = self.q_fc_phi(flat_s)
        return phi

    def generate(self, zs):
        """Map [b,n,k] sized samples of z to [b,n,p] sized images
        """
        # Note that for the purposes of passing through the generator, we need
        # to reshape zs to be size [b*n,k]
        b, n, k = zs.size()
        s = zs.view(b*n, -1)
        # Unflatten
        s = F.relu(self.p_fc_upsample(s)).unsqueeze(2).unsqueeze(3)
        s = F.relu(self.p_deconv_1(s))
        s = F.relu(self.p_deconv_2(s))
        s = self.p_deconv_3(s)
        mu_xs = s.view(b, n, -1)
        return mu_xs

    def forward(self, x):
        # VAE.forward() is not used for training, but we'll treat it like a
        # classic autoencoder by taking a single sample of z ~ q
        phi = self.infer(x)
        zs = rsample(phi, 1)
        return self.generate(zs).view(x.size())

    def elbo(self, x, n=1):
        """Run input end to end through the VAE and compute the ELBO using n
        samples of z
        """
        phi = self.infer(x)
        zs = rsample(phi, n)
        mu_xs = self.generate(zs)
        return log_p_x(x, mu_xs, self.log_sig_x.exp()) - kl_q_p(zs, phi)
    
# @title VAE Helper Functions

def kl_q_p(zs, phi):
    """Given [b,n,k] samples of z drawn from q, compute estimate of KL(q||p).
    phi must be size [b,k+1]

    This uses mu_p = 0 and sigma_p = 1, which simplifies the log(p(zs)) term to
    just -1/2*(zs**2)
    """
    b, n, k = zs.size()
    mu_q, log_sig_q = phi[:,:-1], phi[:,-1]
    log_p = -0.5*(zs**2)
    log_q = -0.5*(zs - mu_q.view(b,1,k))**2 / log_sig_q.exp().view(b,1,1)**2 - log_sig_q.view(b,1,-1)
    # Size of log_q and log_p is [b,n,k]. Sum along [k] but mean along [b,n]
    return (log_q - log_p).sum(dim=2).mean(dim=(0,1))

def log_p_x(x, mu_xs, sig_x):
    """Given [batch, ...] input x and [batch, n, ...] reconstructions, compute
    pixel-wise log Gaussian probability

    Sum over pixel dimensions, but mean over batch and samples.
    """
    b, n = mu_xs.size()[:2]
    # Flatten out pixels and add a singleton dimension [1] so that x will be
    # implicitly expanded when combined with mu_xs
    x = x.reshape(b, 1, -1)
    _, _, p = x.size()
    squared_error = (x - mu_xs.view(b, n, -1))**2 / (2*sig_x**2)

    # Size of squared_error is [b,n,p]. log prob is by definition sum over [p].
    # Expected value requires mean over [n]. Handling different size batches
    # requires mean over [b].
    return -(squared_error + torch.log(sig_x)).sum(dim=2).mean(dim=(0,1))

def rsample(phi, n_samples):
    """Sample z ~ q(z;phi)
    Ouput z is size [b,n_samples,K] given phi with shape [b,K+1]. The first K
    entries of each row of phi are the mean of q, and phi[:,-1] is the log
    standard deviation
    """
    b, kplus1 = phi.size()
    k = kplus1-1
    mu, sig = phi[:, :-1], phi[:,-1].exp()
    eps = torch.randn(b, n_samples, k, device=phi.device)
    return eps*sig.view(b,1,1) + mu.view(b,1,k)