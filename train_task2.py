import os
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from model import *
from utils import *


def task2(device):
    batch_size = 128
    num_workers = 4
    num_epochs = 50

    tensorboard_log_path = os.path.join('tb_log', 'task2')
    checkpoint_path = os.path.join('checkpoint', 'task2')
    os.makedirs(tensorboard_log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    writer = SummaryWriter(tensorboard_log_path)

    task2_train_dataset = Task2TrainDataset('./data/task2_dataset_kotae.csv')
    task2_train_dataloader = DataLoader(task2_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    st_embedding = SpatioTemporalEmbedding().to(device)
    encoding = Encoding().to(device)
    decoding = Decoding().to(device)
    optimizer = torch.optim.AdamW(list(st_embedding.parameters())+list(encoding.parameters())+list(decoding.parameters()), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch_id in range(num_epochs):
        for batch_id, batch in enumerate(tqdm(task2_train_dataloader)):
            day_of_week, time_of_day, location_x, location_y = batch['day_of_week'], batch['time_of_day'], batch['location_x'], batch['location_y']

            day_of_week = day_of_week.to(device)
            time_of_day = time_of_day.to(device)
            location_x = location_x.to(device)
            location_y = location_y.to(device)

            encoding_day_of_week = day_of_week[:, :16]
            encoding_time_of_day = time_of_day[:, :16]
            encoding_location_x = location_x[:, :16]
            encoding_location_y = location_y[:, :16]

            decoding_day_of_week = day_of_week[:, 16:]
            decoding_time_of_day = time_of_day[:, 16:]
            batch_size = day_of_week.size(0)
            zeros_tensor = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)
            decoding_input_location_x = torch.cat((zeros_tensor, location_x[:, 16:-1]), dim=1)
            decoding_input_location_y = torch.cat((zeros_tensor, location_y[:, 16:-1]), dim=1)
            decoding_label_location_x = location_x[:, 16:]
            decoding_label_location_y = location_y[:, 16:]

            encoding_input_embed = st_embedding(encoding_day_of_week, encoding_time_of_day, encoding_location_x, encoding_location_y)
            decoding_input_embed = st_embedding(decoding_day_of_week, decoding_time_of_day, decoding_input_location_x, decoding_input_location_y)
            h = encoding(encoding_input_embed)
            pred_x, pred_y, _ = decoding(decoding_input_embed, h)

            loss =( criterion(pred_x.reshape(-1, 200), decoding_label_location_x.reshape(-1)-1) + criterion(pred_y.reshape(-1, 200), decoding_label_location_y.reshape(-1)-1))/2
            loss.backward()
            nn.utils.clip_grad_norm_(list(st_embedding.parameters())+list(encoding.parameters())+list(decoding.parameters()), max_norm=5, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

            step = epoch_id * len(task2_train_dataloader) + batch_id
            writer.add_scalar('loss', loss.detach().item(), step)

    torch.save(st_embedding.state_dict(), os.path.join(checkpoint_path, 'st_embedding.pth'))
    torch.save(encoding.state_dict(), os.path.join(checkpoint_path, 'encoding.pth'))
    torch.save(decoding.state_dict(), os.path.join(checkpoint_path, 'decoding.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')
    set_random_seed(args.seed)

    task2(device)
