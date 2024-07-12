import torch, torch.nn as nn

HID = 256

class Encoder(nn.Module):
    def __init__(self, input_channels, D):
        super(Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, HID, 4, stride=2, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),
            nn.Conv2d(HID, HID, 4, stride=2, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(HID, HID, 3, stride=1, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(HID, HID, 1, stride=1, padding=0),
            nn.BatchNorm2d(HID),
            nn.ReLU(),  
        )

        self.proj = nn.Conv2d(HID, D, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_block(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = self.proj(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_channels, D):
        super(Decoder, self).__init__()
        self.res1 = nn.Sequential(
            nn.Conv2d(HID, HID, 3, stride=1, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(HID, HID, 3, stride=1, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),  
        )

        self.convtrans_block = nn.Sequential(
            nn.ConvTranspose2d(HID, HID, 4, stride=2, padding=1),
            nn.BatchNorm2d(HID),
            nn.ReLU(),
            nn.ConvTranspose2d(HID, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(D, HID, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = self.convtrans_block(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.K, self.D = config["K"], config["D"]
        self.encoder = Encoder(config["channels"], self.D)
        self.embedding = nn.Embedding(num_embeddings=self.K, embedding_dim=self.D)
        self.decoder = Decoder(config["channels"], self.D)

    def forward(self, x):
        enc = self.encoder(x)

        B, C, H, W = enc.shape
        quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        embed_expanded = self.embedding.weight.view(1, self.K, self.D).expand(B, self.K, self.D)
        dists = torch.cdist(quant_input, embed_expanded)
        closest = torch.argmin(dists, dim=-1)
        quantized = self.embedding(closest)

        enc = enc.view(B, -1, self.D)
        # losses
        commitment_loss = torch.mean((quantized.detach() - enc)**2)
        codebook_loss = torch.mean((quantized - enc.detach())**2)
        quantize_loss = codebook_loss + 0.255555 * commitment_loss

        # quant_out trick to get gradients to the encoder
        quant_out = enc + (quantized - enc).detach()
        quant_out = quant_out.view(B, C, H, W)
        output = self.decoder(quant_out)

        return output, closest, quantize_loss
