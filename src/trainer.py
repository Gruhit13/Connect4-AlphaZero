import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import tqdm
import numpy as np

from model import Model
from buffer import Buffer


class Trainer:
    def __init__(self, model: Model, buffer: Buffer, base_lr:float = 0.001,
                 weight_decay=1e-4, device:str='cpu'):
        self.main_model = model
        self.main_buffer = buffer
        self.global_step = 0
        self.device = device

        # optimizer
        self.optimizer = optim.SGD(
            self.main_model.parameters(),
            lr = base_lr,
            weight_decay = weight_decay,
            momentum = 0.9
        )

        # self.scheduler = optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     base_lr = base_lr,
        #     max_lr = 0.1
        # )

        # Tensorboard summary writer
        self.writer = SummaryWriter()

    def transfer_buffer(self, buffer) -> None:
        for state, value, policy in zip(buffer.state, buffer.value, buffer.policy):
            self.main_buffer.store_experience(
                state = state,
                value = value,
                policy = policy
            )

    def reset_buffer(self) -> None:
        self.main_buffer.reset()

    # learn from the buffer
    def learn(self, state: np.ndarray, value: np.ndarray, policy: np.ndarray) -> float:

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        value = torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(-1)
        policy = torch.tensor(policy, dtype=torch.float32, device=self.device)

        pred_val, pred_policy = self.main_model(state)

        self.optimizer.zero_grad()
        loss = self.main_model.get_loss(pred_val, pred_policy, value, policy)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    # Training loop for the model
    def train_model(self, epochs: int, batch_size: int):

        train_steps = np.ceil(len(self.main_buffer) / batch_size).astype(np.int32)

        # perform the training
        for epoch in range(epochs):
            for state, value, policy in tqdm(self.main_buffer.sample(batch_size), total=train_steps, desc=f'Epoch:{epoch+1}'):
                loss = self.learn(state, value, policy)
                self.writer.add_scalar("loss", loss, self.global_step)
                self.global_step += 1

        self.writer.flush()

    # close the writer
    def close_writer(self):
        self.writer.close()

    # Save the model
    def save_model(self, step: int):
        torch.save(self.main_model.state_dict(), f'TargetModel_{step}.pt')
        torch.save(self.optimizer.state_dict(), f'Optimizer_{step}.pt')