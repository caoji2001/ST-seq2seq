from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Task1TrainDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

        self.sequences_day_of_week = []
        self.sequences_time_of_day = []
        self.sequences_location_x = []
        self.sequences_location_y = []
        sequence_length = 32

        grouped = df.groupby('uid')
        for uid, group in tqdm(grouped):
            if uid >= 80000:
                group = group[group['d'] < 60]

            day_of_week = group['d'].to_numpy() % 7
            time_of_day = group['t'].to_numpy()
            location_x = group['x'].to_numpy()
            location_y = group['y'].to_numpy()

            seq_count = len(day_of_week) // sequence_length
            for i in range(seq_count):
                start = i * sequence_length
                end = (i + 1) * sequence_length

                self.sequences_day_of_week.append(day_of_week[start:end])
                self.sequences_time_of_day.append(time_of_day[start:end])
                self.sequences_location_x.append(location_x[start:end])
                self.sequences_location_y.append(location_y[start:end])

        self.sequences_day_of_week = np.vstack(self.sequences_day_of_week)
        self.sequences_time_of_day = np.vstack(self.sequences_time_of_day)
        self.sequences_location_x = np.vstack(self.sequences_location_x)
        self.sequences_location_y = np.vstack(self.sequences_location_y)

    def __len__(self):
        return len(self.sequences_day_of_week)
    
    def __getitem__(self, index):
        return {
            'day_of_week': torch.tensor(self.sequences_day_of_week[index], dtype=torch.int64),
            'time_of_day': torch.tensor(self.sequences_time_of_day[index], dtype=torch.int64),
            'location_x': torch.tensor(self.sequences_location_x[index], dtype=torch.int64),
            'location_y': torch.tensor(self.sequences_location_y[index], dtype=torch.int64),
        }


class Task2TrainDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

        self.sequences_day_of_week = []
        self.sequences_time_of_day = []
        self.sequences_location_x = []
        self.sequences_location_y = []
        sequence_length = 32

        grouped = df.groupby('uid')
        for uid, group in tqdm(grouped):
            if uid >= 22500:
                group = group[group['d'] < 60]

            day_of_week = group['d'].to_numpy() % 7
            time_of_day = group['t'].to_numpy()
            location_x = group['x'].to_numpy()
            location_y = group['y'].to_numpy()

            seq_count = len(day_of_week) // sequence_length
            for i in range(seq_count):
                start = i * sequence_length
                end = (i + 1) * sequence_length

                self.sequences_day_of_week.append(day_of_week[start:end])
                self.sequences_time_of_day.append(time_of_day[start:end])
                self.sequences_location_x.append(location_x[start:end])
                self.sequences_location_y.append(location_y[start:end])

        self.sequences_day_of_week = np.vstack(self.sequences_day_of_week)
        self.sequences_time_of_day = np.vstack(self.sequences_time_of_day)
        self.sequences_location_x = np.vstack(self.sequences_location_x)
        self.sequences_location_y = np.vstack(self.sequences_location_y)

    def __len__(self):
        return len(self.sequences_day_of_week)
    
    def __getitem__(self, index):
        return {
            'day_of_week': torch.tensor(self.sequences_day_of_week[index], dtype=torch.int64),
            'time_of_day': torch.tensor(self.sequences_time_of_day[index], dtype=torch.int64),
            'location_x': torch.tensor(self.sequences_location_x[index], dtype=torch.int64),
            'location_y': torch.tensor(self.sequences_location_y[index], dtype=torch.int64),
        }


class Task1ValDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df = df[df['uid'] >= 80000]

        self.encoding_day_of_week = []
        self.encoding_time_of_day = []
        self.encoding_location_x = []
        self.encoding_location_y = []
        self.decoding_day = []
        self.decoding_time_of_day = []
        self.label_location_x = []
        self.label_location_y = []

        grouped = df.groupby('uid')
        for uid, group in tqdm(grouped):
            group1 = group[group['d'] < 60]
            group2 = group[group['d'] >= 60]

            self.encoding_day_of_week.append(group1['d'].tail(16).values % 7)
            self.encoding_time_of_day.append(group1['t'].tail(16).values)
            self.encoding_location_x.append(group1['x'].tail(16).values)
            self.encoding_location_y.append(group1['y'].tail(16).values)
            self.decoding_day.append(group2['d'].to_numpy())
            self.decoding_time_of_day.append(group2['t'].to_numpy())
            self.label_location_x.append(group2['x'].to_numpy())
            self.label_location_y.append(group2['y'].to_numpy())

    def __len__(self):
        return len(self.encoding_day_of_week)
    
    def __getitem__(self, index):
        return {
            'encoding_day_of_week': torch.tensor(self.encoding_day_of_week[index], dtype=torch.int64),
            'encoding_time_of_day': torch.tensor(self.encoding_time_of_day[index], dtype=torch.int64),
            'encoding_location_x': torch.tensor(self.encoding_location_x[index], dtype=torch.int64),
            'encoding_location_y': torch.tensor(self.encoding_location_y[index], dtype=torch.int64),
            'decoding_day': torch.tensor(self.decoding_day[index], dtype=torch.int64),
            'decoding_time_of_day': torch.tensor(self.decoding_time_of_day[index], dtype=torch.int64),
            'label_location_x': torch.tensor(self.label_location_x[index], dtype=torch.int64),
            'label_location_y': torch.tensor(self.label_location_y[index], dtype=torch.int64)
        }


class Task2ValDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df = df[df['uid'] >= 22500]

        self.encoding_day_of_week = []
        self.encoding_time_of_day = []
        self.encoding_location_x = []
        self.encoding_location_y = []
        self.decoding_day = []
        self.decoding_time_of_day = []
        self.label_location_x = []
        self.label_location_y = []

        grouped = df.groupby('uid')
        for uid, group in tqdm(grouped):
            group1 = group[group['d'] < 60]
            group2 = group[group['d'] >= 60]

            self.encoding_day_of_week.append(group1['d'].tail(16).values % 7)
            self.encoding_time_of_day.append(group1['t'].tail(16).values)
            self.encoding_location_x.append(group1['x'].tail(16).values)
            self.encoding_location_y.append(group1['y'].tail(16).values)
            self.decoding_day.append(group2['d'].to_numpy())
            self.decoding_time_of_day.append(group2['t'].to_numpy())
            self.label_location_x.append(group2['x'].to_numpy())
            self.label_location_y.append(group2['y'].to_numpy())

    def __len__(self):
        return len(self.encoding_day_of_week)
    
    def __getitem__(self, index):
        return {
            'encoding_day_of_week': torch.tensor(self.encoding_day_of_week[index], dtype=torch.int64),
            'encoding_time_of_day': torch.tensor(self.encoding_time_of_day[index], dtype=torch.int64),
            'encoding_location_x': torch.tensor(self.encoding_location_x[index], dtype=torch.int64),
            'encoding_location_y': torch.tensor(self.encoding_location_y[index], dtype=torch.int64),
            'decoding_day': torch.tensor(self.decoding_day[index], dtype=torch.int64),
            'decoding_time_of_day': torch.tensor(self.decoding_time_of_day[index], dtype=torch.int64),
            'label_location_x': torch.tensor(self.label_location_x[index], dtype=torch.int64),
            'label_location_y': torch.tensor(self.label_location_y[index], dtype=torch.int64)
        }
