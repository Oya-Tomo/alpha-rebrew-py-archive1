import os
import torch
from torch.utils.data import DataLoader
import ray

from data import MCTSDataset
from match import auto_match, MCTStep
from model import PVNet


def train():
    ray.init(num_cpus=10, num_gpus=1)

    loops = 10000
    gpl = 300
    sim = 30
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PVNet().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    loss_history = []

    for loop in range(loops):
        print(f"Loop: {loop}")
        weights = ray.put(net.cpu().state_dict())

        print("Auto matching...")
        matches = [auto_match.remote(weights, sim) for _ in range(gpl)]
        results: list[MCTStep] = [ray.get(match) for match in matches]

        print("Creating dataset...")
        dataset = MCTSDataset()
        for result in results:
            dataset.add(result)

        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        device = next(net.parameters()).device

        print("Training...")
        for epoch in range(epochs):
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

            print(f"Loss: {train_loss}")
            loss_history.append(train_loss)

        if loop % 10 == 99:
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
