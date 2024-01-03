import torch
from torch import nn


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self):
        super(SpatioTemporalEmbedding, self).__init__()

        self.day_of_week_embedding = nn.Embedding(7, 32)
        self.time_of_day_embedding = nn.Embedding(48, 32)
        self.location_x_embedding = nn.Embedding(201, 128)
        self.location_y_embedding = nn.Embedding(201, 128)

    def forward(self, day, time, location_x, location_y):
        day_embed = self.day_of_week_embedding(day)
        time_embed = self.time_of_day_embedding(time)
        location_x_embed = self.location_x_embedding(location_x)
        location_y_embed = self.location_y_embedding(location_y)

        embed = torch.cat((day_embed, time_embed, location_x_embed, location_y_embed), dim=-1)
        return embed


class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()

        self.gru = nn.GRU(input_size=32+32+128+128, hidden_size=256, num_layers=4, dropout=0.1)

    def forward(self, input):
        if input.dim() == 3:
            input = input.permute(1, 0, 2)
        out, h = self.gru(input)
        return h


class Decoding(nn.Module):
    def __init__(self):
        super(Decoding, self).__init__()

        self.gru = nn.GRU(input_size=32+32+128+128, hidden_size=256, num_layers=4, dropout=0.1)
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 200)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 200)
        )

    def forward(self, input, h0):
        if input.dim() == 3:
            input = input.permute(1, 0, 2)
        out, h = self.gru(input, h0)
        if out.dim() == 3:
            out = out.permute(1, 0, 2)

        pred_x = self.mlp1(out)
        pred_y = self.mlp2(out)

        return pred_x, pred_y, h
