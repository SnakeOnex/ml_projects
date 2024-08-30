import torch, torch.nn as nn, torchvision.models as models
from dataclasses import dataclass
from collections import namedtuple
from torch.nn import functional as F
from einops import rearrange, einsum

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

class STQuantize(nn.Module):
    # def __init__(self, code_dim, num_codes, commitment_cost=0.25):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.code_dim = config.D
        self.num_codes = config.K
        self.commitment_cost = 0.25
        self.embedding = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.kaiming_uniform_(self.embedding.weight.data)

    def forward(self, z):
        B, C, H, W = z.shape
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        dist = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * einsum(
                z_flattened,
                rearrange(self.embedding.weight, "n d -> d n"),
                "b d, d n -> b n",
            )
        )

        min_encoding_indices = torch.argmin(dist, dim=1)

        encodings = F.one_hot(min_encoding_indices, self.num_codes).type(z.dtype)
        avg_use = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_use * torch.log(avg_use + 1e-10)))

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = (
            self.commitment_cost * ((z_q.detach() - z) ** 2).mean()
            + ((z_q - z.detach()) ** 2).mean()
        )

        z_q = (
            z + (z_q - z).detach()
        )  # Straight-through estimator, must be done after calculating the loss!

        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        min_encoding_indices = rearrange(
            min_encoding_indices, "(b h w) -> b h w", **{"b": B, "h": H, "w": W}
        )
        return z_q, loss, (min_encoding_indices, perplexity)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        z_q = z_q.view(shape)
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q

class Quantize(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(Quantize, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=self.config.K, embedding_dim=self.config.D)

    def forward(self, enc):
        B, C, H, W = enc.shape
        quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        embed_expanded = self.embedding.weight.view(1, self.config.K, self.config.D).expand(B, self.config.K, self.config.D)
        dists = torch.cdist(quant_input, embed_expanded)
        closest = torch.argmin(dists, dim=-1)
        quantized = self.embedding(closest)
        enc = enc.view(B, -1, self.config.D)

        commitment_loss = torch.mean((quantized.detach() - enc)**2)
        codebook_loss = torch.mean((quantized - enc.detach())**2)
        quantize_loss = codebook_loss + 1.0 * commitment_loss

        # quant_out trick to get gradients to the encoder
        quant_out = enc + (quantized - enc).detach()
        quant_out = quant_out.view(B, C, H, W)
        return quant_out, quantize_loss, (closest, 0)


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(VQVAE, self).__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.embedding = nn.Embedding(num_embeddings=self.config.K, embedding_dim=self.config.D)
        # self.embedding.weight.data.uniform_(-1/self.config.K, 1/self.config.K)
        self.decoder = Decoder(self.config)
        # self.quantize = STQuantize(self.config.D, self.config.K)
        self.quantize = Quantize(self.config)

    def decode(self, z):
        z = torch.clamp(z, 0, self.config.K-1)

        quantized = self.embedding(z)
        quantized = quantized.view(1, self.config.D, self.config.image_sz//self.config.downsample_factor, self.config.image_sz//self.config.downsample_factor)
        return self.decoder(quantized)

    def forward(self, x, verbose=False):
        enc = self.encoder(x)

        if verbose: print("Input shape:", x.shape) 
        if verbose: print("Encoder output shape:", enc.shape) 

        B, C, H, W = enc.shape
        # quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        # embed_expanded = self.embedding.weight.view(1, self.config.K, self.config.D).expand(B, self.config.K, self.config.D)
        # dists = torch.cdist(quant_input, embed_expanded)
        # closest = torch.argmin(dists, dim=-1)
        # quantized = self.embedding(closest)
        # print("Quantized shape:", quantized.shape)

        quantized, quantize_loss, (closest, perplexity) = self.quantize(enc)
        # print("Quantized2 shape:", quantized.shape)

        if verbose: print("Quantized shape:", closest.shape) 
        if verbose: print("Quantized shape:", closest.view(-1)) 

        enc = enc.view(B, -1, self.config.D)
        # losses
        # commitment_loss = torch.mean((quantized.detach() - enc)**2)
        # codebook_loss = torch.mean((quantized - enc.detach())**2)
        # quantize_loss = codebook_loss + 1.0 * commitment_loss

        # quant_out trick to get gradients to the encoder
        # quant_out = enc + (quantized - enc).detach()
        # quant_out = quant_out.view(B, C, H, W)
        # output = self.decoder(quant_out)
        output = self.decoder(quantized)
        # exit(0)
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"

        return {"output": output, "closest": closest, "quantize_loss": quantize_loss}

# class PerceptualLoss(nn.Module):
    # def __init__(self):
        # super(PerceptualLoss, self).__init__()
        # vgg = models.vgg16(pretrained=True).features
        # self.slice1 = nn.Sequential(*vgg[:4])
        # self.slice2 = nn.Sequential(*vgg[4:9])
        # self.slice3 = nn.Sequential(*vgg[9:16])
        # self.slice4 = nn.Sequential(*vgg[16:23])
        # for param in self.parameters():
            # param.requires_grad = False

    # def forward(self, x, y):
        # x = self.slice1(x)
        # y = self.slice1(y)
        # loss = torch.nn.functional.l1_loss(x, y)

        # x = self.slice2(x)
        # y = self.slice2(y)
        # loss += torch.nn.functional.l1_loss(x, y)

        # x = self.slice3(x)
        # y = self.slice3(y)
        # loss += torch.nn.functional.l1_loss(x, y)

        # x = self.slice4(x)
        # y = self.slice4(y)
        # loss += torch.nn.functional.l1_loss(x, y)

        # return loss

# class VQLPIPS(nn.Module):
# class PerceptualLoss(nn.Module):
    # def __init__(
        # self, reconstruction_weight=1.0, codebook_weight=1.0, perceptual_weight=0.1
    # ):
        # super().__init__()
        # self.perceptual_loss_fn = LPIPS()
        # self.reconstruction_weight = reconstruction_weight
        # self.codebook_weight = codebook_weight
        # self.perceptual_weight = perceptual_weight

    # # def forward(self, x_recon, x_orig, codebook_loss):
    # def forward(self, x_recon, x_orig):
        # reconstruction_loss = (x_recon.contiguous() - x_orig.contiguous()).abs().mean()
        # perceptual_loss = self.perceptual_loss_fn(
            # x_recon.clamp(-1.0, 1.0), x_orig
        # ).mean()
        # loss = (
            # reconstruction_loss * self.reconstruction_weight
            # + perceptual_loss * self.perceptual_weight
            # # + codebook_loss * self.codebook_weight
        # )
        # loss_items = {
            # "loss": loss.item(),
            # "reconstruction_loss": reconstruction_loss.item(),
            # "perceptual_loss": perceptual_loss.item(),
            # # "codebook_loss": codebook_loss.item(),
        # }
        # return loss, loss_items


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features

        self.net = vgg16(pretrained=True, requires_grad=False)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        # self.lins = nn.ModuleList(self.lins)

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = "vgg.pth"
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))

        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        feats0, feats1, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


from blocks import Decoder as DecoderGAN, Encoder as EncoderGAN
from omegaconf import OmegaConf

class VQGAN(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super(VQGAN, self).__init__()
        self.config = config
        conf = OmegaConf.load("../vqgan/src/configs/ffhq_default.yaml")
        self.encoder = EncoderGAN(**conf["autoencoder"])
        self.decoder = DecoderGAN(**conf["autoencoder"])
        self.quantize = STQuantize(self.config)

        # self.quant_conv = nn.Conv2d(conf["autoencoder"]["z_channels"], conf["vector_quantization"]["params"]["code_dim"], 1)
        self.quant_conv = nn.Conv2d(conf["autoencoder"]["z_channels"], 64, 1)
        self.post_quant_conv = nn.Conv2d(64, conf["autoencoder"]["z_channels"], 1)

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.quant_conv(enc)
        quantized, quantize_loss, (closest, perplexity) = self.quantize(enc)
        quantized = self.post_quant_conv(quantized)
        output = self.decoder(quantized)
        return {"output": output, "closest": closest, "quantize_loss": quantize_loss}
