import torch
import torch.nn as nn

import torchvision

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()

        self.codebook_size = 8
        self.codebook_dim = 4

        # B x 1 x 28 x 28 input
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, self.codebook_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(self.codebook_dim),
                nn.ReLU(),
                nn.Conv2d(self.codebook_dim, self.codebook_dim, 1, stride=1, padding=0),
                nn.BatchNorm2d(self.codebook_dim),
                nn.ReLU(),
                nn.Conv2d(self.codebook_dim, self.codebook_dim, 1, stride=1, padding=0),
                nn.BatchNorm2d(self.codebook_dim),
                nn.ReLU(),
        )
        # B x 4 x 28 x 28 output
        self.embedding = nn.Embedding(num_embeddings=self.codebook_size, embedding_dim=self.codebook_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.codebook_dim, self.codebook_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.codebook_dim),
            nn.ReLU(),
            nn.Conv2d(self.codebook_dim, self.codebook_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.codebook_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(self.codebook_dim, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        print(enc.shape)

        B, C, H, W = enc.shape
        quant_input = enc.view(B, -1, C) # reshape to be B x -1 x C
        embed_expanded = self.embedding.weight.view(1, self.codebook_size, self.codebook_dim).expand(B, self.codebook_size, self.codebook_dim)
        dists = torch.cdist(quant_input, embed_expanded)
        closest = torch.argmin(dists, dim=-1)
        quantized = self.embedding(closest)

        enc = enc.view(B, -1, self.codebook_dim)
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
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # take subset
    # mnist = torch.utils.data.Subset(mnist, range(1600))

    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    model = VQVAE()
    # model.load_state_dict(torch.load("vqgan.pth"))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    for epoch in range(5):
        for x, y in train_loader:
            optim.zero_grad()
            out, _, loss = model(x)

            reconstruct_loss = crit(out, x)
            loss = reconstruct_loss + loss

            loss.backward()
            optim.step()
            print(f"e={epoch}: loss={loss.item():.4f}")

    # # # save model
    torch.save(model.state_dict(), "vqgan.pth")

    # load model
    model = VQVAE()
    model.load_state_dict(torch.load("vqgan.pth"))
    model.eval()

    idxs = torch.randint(0, len(test_mnist), (16,))
    print(idxs)
    images_gt = torch.stack([test_mnist[i][0] for i in idxs])

    images_pred, quantized, _ = model(images_gt)
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

    import matplotlib.pyplot as plt
    plt.imshow(grid_final)
    plt.show()


