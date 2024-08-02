import torch, torch.nn as nn, torchvision.models as models
from dataclasses import dataclass

@dataclass
class VQVAEConfig:
    in_channels: int = 3
    image_sz: int = 32
    ch_base: int = 32
    ch_mult: tuple[int] = (1, 2)
    K: int = 512
    D: int = 64

    @property
    def num_resolutions(self):
        return len(self.ch_mult)

    @property
    def downsample_factor(self):
        return 2**(self.num_resolutions-1)

    @property
    def final_channels(self):
        return self.ch_base * self.ch_mult[-1]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x) + x

class Encoder(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(Encoder, self).__init__()
        self.config = config

        self.conv_in = torch.nn.Conv2d(self.config.in_channels, self.config.ch_base, kernel_size=3, stride=1, padding=1)

        self.downsample = nn.ModuleList()
        for i in range(self.config.num_resolutions-1):
            self.downsample.append(nn.Sequential(
                ResBlock(self.config.ch_base * self.config.ch_mult[i], self.config.ch_base * self.config.ch_mult[i]),
                nn.Conv2d(self.config.ch_base * self.config.ch_mult[i], self.config.ch_base * self.config.ch_mult[i+1], kernel_size=4, stride=2, padding=1),
            ))

        self.res_layer = ResBlock(self.config.final_channels, self.config.final_channels)

        self.proj = nn.Conv2d(self.config.final_channels, self.config.D, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.downsample: x = layer(x)
        x = self.res_layer(x)
        return self.proj(x)

class Decoder(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(Decoder, self).__init__()
        self.config = config

        self.res_layer = ResBlock(self.config.final_channels, self.config.final_channels)

        self.upsample = nn.ModuleList()
        for i in range(self.config.num_resolutions-1, 0, -1):
            self.upsample.append(nn.Sequential(
                ResBlock(self.config.ch_base * self.config.ch_mult[i], self.config.ch_base * self.config.ch_mult[i]),
                nn.ConvTranspose2d(self.config.ch_base * self.config.ch_mult[i], self.config.ch_base * self.config.ch_mult[i-1], kernel_size=4, stride=2, padding=1),
            ))
        self.proj = nn.Conv2d(self.config.D, self.config.final_channels, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(self.config.ch_base, self.config.in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.res_layer(x)
        for layer in self.upsample: x = layer(x)
        x = self.conv_out(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(VQVAE, self).__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.embedding = nn.Embedding(num_embeddings=self.config.K, embedding_dim=self.config.D)
        self.embedding.weight.data.uniform_(-1/self.config.K, 1/self.config.K)
        self.decoder = Decoder(self.config)

    def decode(self, z):
        z = torch.clamp(z, 0, self.config.K-1)

        quantized = self.embedding(z)
        # print("Quantized shape:", quantized.shape)
        quantized = quantized.view(1, self.config.D, self.config.image_sz//self.config.downsample_factor, self.config.image_sz//self.config.downsample_factor)
        return self.decoder(quantized)

    def forward(self, x, verbose=False):
        enc = self.encoder(x)

        if verbose: print("Input shape:", x.shape) 
        if verbose: print("Encoder output shape:", enc.shape) 

        B, C, H, W = enc.shape
        quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        embed_expanded = self.embedding.weight.view(1, self.config.K, self.config.D).expand(B, self.config.K, self.config.D)
        dists = torch.cdist(quant_input, embed_expanded)
        closest = torch.argmin(dists, dim=-1)
        quantized = self.embedding(closest)

        if verbose: print("Quantized shape:", closest.shape) 
        if verbose: print("Quantized shape:", closest.view(-1)) 

        enc = enc.view(B, -1, self.config.D)
        # losses
        commitment_loss = torch.mean((quantized.detach() - enc)**2)
        codebook_loss = torch.mean((quantized - enc.detach())**2)
        quantize_loss = codebook_loss + 0.25 * commitment_loss

        # quant_out trick to get gradients to the encoder
        quant_out = enc + (quantized - enc).detach()
        quant_out = quant_out.view(B, C, H, W)
        output = self.decoder(quant_out)
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"


        return {"output": output, "closest": closest, "quantize_loss": quantize_loss}

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])
        self.slice4 = nn.Sequential(*vgg[16:23])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = self.slice1(x)
        y = self.slice1(y)
        loss = torch.nn.functional.l1_loss(x, y)

        x = self.slice2(x)
        y = self.slice2(y)
        loss += torch.nn.functional.l1_loss(x, y)

        x = self.slice3(x)
        y = self.slice3(y)
        loss += torch.nn.functional.l1_loss(x, y)

        x = self.slice4(x)
        y = self.slice4(y)
        loss += torch.nn.functional.l1_loss(x, y)

        return loss

