import time
import os
import torch
from torch.utils.data import DataLoader
import ray

from data import MCTSDataset
from match import auto_match, MCTStep
from model import PVNet


def train():
    ray.init(num_cpus=15, num_gpus=1)

    loops = 10000
    games = 300
    sim = 40
    epochs = 20

    save_epoch = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PVNet().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    loss_history = []

    weights = ray.put(net.state_dict())
    matches = [auto_match.remote(weights, sim) for _ in range(games)]
    dataset = MCTSDataset(50000, 0.1)

    for loop in range(loops):
        lt = time.time()
        print(f"Loop: {loop}")
        weights = ray.put(net.state_dict())

        print("Get matching...")
        for c in range(games):
            fin, matches = ray.wait(matches, num_returns=1)
            rec = ray.get(fin[0])
            dataset.add([rec])
            matches.extend([auto_match.remote(weights, sim)])
            print(f"\rMatch: {c}")

        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        device = next(net.parameters()).device

        print("Training...")
        for epoch in range(epochs):
            et = time.time()
            print(f"Epoch: {epoch}")

            train_loss = 0.0

            for s, p, v in dataloader:
                s, p, v = s.to(device), p.to(device), v.to(device)

                optimizer.zero_grad()
                p_hat, v_hat = net(s)

                p_loss = policy_criterion(p_hat, p)
                v_loss = value_criterion(v_hat, v)

                l2_alpha = 1e-6
                l2 = torch.tensor(0.0, requires_grad=True)
                for param in net.parameters():
                    l2 = l2 + torch.norm(param) ** 2

                loss = p_loss + v_loss + l2 * l2_alpha

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Loss: {train_loss / len(dataloader)}")
            loss_history.append(train_loss / len(dataloader))

            print(f"Epoch Time: {time.time() - et}")

        print(f"Loop Time: {time.time() - lt}")

        if loop % save_epoch == save_epoch - 1:
            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                f"checkpoint/model_{loop}.pt",
            )


if __name__ == "__main__":
    train()
