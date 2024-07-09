import numpy as np
import torch
from torch.utils.data import Dataset
from model import PVNet
from board import Board, Stone
from match import auto_match, MCTStep


class MCTSDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.buffer: list[MCTStep] = []

    def add(self, record: list[MCTStep]):
        self.buffer.extend(record)

    def __getitem__(self, index):
        item: MCTStep = self.buffer[index]

        return (
            item.board.to_tensor(item.stone),
            torch.tensor(item.policy),
            torch.tensor(item.value),
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    net = PVNet()
    rec = auto_match(net.state_dict(), 40, alpha=0.2)

    dataset = MCTSDataset()
    dataset.add(rec)
