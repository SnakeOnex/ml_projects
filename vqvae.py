import torch, torchvision
import torch.nn as nn
import time
import PIL
import argparse
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def __init__(self):
        super(VQVAE, self).__init__()
        self.channels = 1 if dataset == "mnist" else 3

        # B x 1 x 28 x 28 input
        self.encoder = Encoder(self.channels, D)

        # B x 4 x 28 x 28 output
        self.embedding = nn.Embedding(num_embeddings=K, embedding_dim=D)

        self.decoder = Decoder(self.channels, D)

    def forward(self, x):
        enc = self.encoder(x)

        B, C, H, W = enc.shape
        quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        embed_expanded = self.embedding.weight.view(1, K, D).expand(B, K, D)
        dists = torch.cdist(quant_input, embed_expanded)
        closest = torch.argmin(dists, dim=-1)
        quantized = self.embedding(closest)

        enc = enc.view(B, -1, D)
        # losses
        commitment_loss = torch.mean((quantized.detach() - enc)**2)
        codebook_loss = torch.mean((quantized - enc.detach())**2)
        quantize_loss = codebook_loss + 0.255555 * commitment_loss

        # quant_out trick to get gradients to the encoder
        quant_out = enc + (quantized - enc).detach()
        quant_out = quant_out.view(B, C, H, W)
        output = self.decoder(quant_out)

        return output, closest, quantize_loss

def eval_vqvae(model, loader):
    with torch.no_grad():
        r_loss = 0
        q_loss = 0
        count = 0
        for x, _ in loader:
            x = x.to(device)
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            # print(f"e={epoch}: reconstruct_loss={reconstruct_loss.item():.4f}, quant_loss={loss.item():.4f}")
            r_loss += reconstruct_loss
            q_loss += loss
            count += 1
            loss = reconstruct_loss + loss
    return r_loss/count, q_loss/count

def plot_results(model, dataset, image_count, path="vqvae.png", idxs=None):
    if idxs is None:
        idxs = torch.randint(0, len(dataset), (16,))

    images_gt = torch.stack([dataset[i][0] for i in idxs])

    with torch.inference_mode():
        images_pred, quantized, _ = model(images_gt.to(device))
    images_pred = images_pred.cpu().detach()

    grid_pred = torchvision.utils.make_grid(images_pred, nrow=4)
    grid = torchvision.utils.make_grid(images_gt, nrow=4)
    grid_final = torch.cat([grid, grid_pred], dim=2)

    grid_final = grid_final.permute(1, 2, 0)


    grid_final = grid_final.numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    grid_final.save(f"{path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    dataset = args.dataset

    print(f"training VQ-VAE on the {dataset} dataset")

    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(256),
        # torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])

    if dataset == "mnist":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        K, D = 32, 64
    elif dataset == "cifar":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        K, D = 512, 64
    elif dataset == "imagenet":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(128),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = torchvision.datasets.ImageNet(
                root='/datagrid/public_datasets/imagenet/imagenet_pytorch',
                split='train',
                transform=transform
        )

        train_dataset = torch.utils.data.Subset(train_dataset, range(10_000))

        test_dataset = torchvision.datasets.ImageNet(
                root='/datagrid/public_datasets/imagenet/imagenet_pytorch',
                split='val',
                transform=transform
        )
        test_dataset = torch.utils.data.Subset(test_dataset, range(1_000))
        K, D = 512, 64
    print(f"Dataset size: train_len={len(train_dataset)}, test_len={len(test_dataset)}")
    print(f"K={K}, D={D}")

    Path("results").mkdir(parents=True, exist_ok=True)

    idxs = torch.randint(0, len(test_dataset), (16,))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=32, 
            prefetch_factor=16, 
            pin_memory=True,
            persistent_workers=True
            )
    model = VQVAE().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.MSELoss()

    for epoch in range(100):
        r_loss = 0
        q_loss = 0
        count = 0
        st = time.time()
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            optim.zero_grad()
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            r_loss += reconstruct_loss
            q_loss += loss
            count += 1
            loss = 2 * reconstruct_loss + loss

            loss.backward()
            optim.step()
        r_loss /= count
        q_loss /= count
        val_r_loss, val_q_loss = eval_vqvae(model, train_loader)
        print(f"e={epoch:2}, trn_r_l={r_loss:.4f}, val_r_l={val_r_loss:.4f}, trn_q_l={q_loss:.4f}, val_q_l={val_q_loss:.4f}, t={(time.time() - st):.2f} s")

        if epoch % 1 == 0:
            plot_results(model, test_dataset, 16, path=f"results/vqvae_{epoch}.png", idxs=idxs)

    # save model
    torch.save(model.state_dict(), "vqvae.pth")

    model.load_state_dict(torch.load("vqvae.pth"))

    plot_results(model, test_dataset, 16, path=f"random_sample.png")







