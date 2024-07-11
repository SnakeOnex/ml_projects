import torch
import torch.nn as nn

import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

K = 512
D = 2

dataset = "mnist" # mnist or cifar

class Encoder(nn.Module):
    def __init__(self, input_channels, D):
        super(Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, D, 4, stride=2, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(D, D, 3, stride=1, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(D, D, 1, stride=1, padding=0),
            nn.BatchNorm2d(D),
            nn.ReLU(),  
        )

    def forward(self, x):
        x = self.conv_block(x)
        # x = x + self.res1(x)
        # x = x + self.res2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_channels, D):
        super(Decoder, self).__init__()
        self.res1 = nn.Sequential(
            nn.Conv2d(D, D, 3, stride=1, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(D, D, 3, stride=1, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),  
        )

        self.convtrans_block = nn.Sequential(
            nn.ConvTranspose2d(D, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = x + self.res1(x)
        # x = x + self.res2(x)
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

        self.decoder = nn.Sequential(
            nn.Conv2d(D, D, 1, stride=1, padding=0),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.Conv2d(D, D, 1, stride=1, padding=0),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.ConvTranspose2d(D, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        # self.decoder = Decoder(self.channels, D)

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
        quantize_loss = codebook_loss + 0.2 * commitment_loss

        # quant_out trick to get gradients to the encoder
        quant_out = enc + (quantized - enc).detach()
        quant_out = quant_out.view(B, C, H, W)
        output = self.decoder(quant_out)

        return output, closest, quantize_loss


if __name__ == "__main__":
    print(f"training VQ-VAE on the {dataset} dataset")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Resize((32, 32)),
    ])

    if dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == "cifar":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # take subset
    # mnist = torch.utils.data.Subset(mnist, range(1600))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    model = VQVAE().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    for epoch in range(5):
        for x, _ in train_loader:
            x = x.to(device)
            optim.zero_grad()
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            loss = reconstruct_loss + loss

            loss.backward()
            optim.step()
            print(f"e={epoch}: loss={loss.item():.4f}")

    # # save model
    torch.save(model.state_dict(), "vqvae.pth")

    # load model
    model.load_state_dict(torch.load("vqvae.pth"))
    model.eval()

    idxs = torch.randint(0, len(test_dataset), (16,))
    print(idxs)
    images_gt = torch.stack([test_dataset[i][0] for i in idxs])

    images_pred, quantized, _ = model(images_gt.to(device))
    images_pred = images_pred.cpu().detach()
    print(quantized)

    print("closest: ", quantized.shape)

    print(images_gt.min(), images_gt.max())
    print(images_pred.min(), images_pred.max())

    grid_pred = torchvision.utils.make_grid(images_pred, nrow=4)

    # grid_pred = grid_pred.permute(1, 2, 0)
    grid = torchvision.utils.make_grid(images_gt, nrow=4)
    # grid = grid.permute(1, 2, 0)

    grid_final = torch.cat([grid, grid_pred], dim=2)

    grid_final = grid_final.permute(1, 2, 0)

    import PIL

    grid_final = grid_final.numpy()
    grid_final = (grid_final * 255).astype("uint8")
    grid_final = PIL.Image.fromarray(grid_final)

    # save
    grid_final.save("vqvae.png")


    # import matplotlib.pyplot as plt
    # plt.imshow(grid_final)
    # plt.show()


