import os
import argparse
import json
from tqdm import tqdm

import torch
from dataset import *
from model import *


def task2(st_embedding_pth, encoding_pth, decoding_pth, device):
    result_path = os.path.join('result', 'task2')
    os.makedirs(result_path, exist_ok=True)

    task2_val_dataset = Task2ValDataset('./data/task2_dataset_kotae.csv')

    st_embedding = SpatioTemporalEmbedding().to(device)
    encoding = Encoding().to(device)
    decoding = Decoding().to(device)

    st_embedding.load_state_dict(torch.load(st_embedding_pth, map_location=device))
    encoding.load_state_dict(torch.load(encoding_pth, map_location=device))
    decoding.load_state_dict(torch.load(decoding_pth, map_location=device))

    result = dict()
    result['generated'] = []
    result['reference'] = []
    st_embedding.eval()
    encoding.eval()
    decoding.eval()

    with torch.no_grad():
        for data in tqdm(task2_val_dataset):
            encoding_day_of_week = data['encoding_day_of_week'].to(device)
            encoding_time_of_day = data['encoding_time_of_day'].to(device)
            encoding_location_x = data['encoding_location_x'].to(device)
            encoding_location_y = data['encoding_location_y'].to(device)
            decoding_day = data['decoding_day'].to(device)
            decoding_time_of_day = data['decoding_time_of_day'].to(device)
            label_location_x = data['label_location_x'].to(device)
            label_location_y = data['label_location_y'].to(device)

            encoding_input_embed = st_embedding(encoding_day_of_week, encoding_time_of_day, encoding_location_x, encoding_location_y)
            h = encoding(encoding_input_embed)

            pred_len = decoding_day.size(0)
            pred_x_array = torch.zeros((pred_len, ), dtype=torch.int64, device=device)
            pred_y_array = torch.zeros((pred_len, ), dtype=torch.int64, device=device)

            for pred_step in range(pred_len):
                decoding_input_day_of_week = decoding_day[pred_step].unsqueeze(0) % 7
                decoding_input_time_of_day = decoding_time_of_day[pred_step].unsqueeze(0)

                if pred_step == 0:
                    decoding_input_location_x = torch.zeros((1, ), dtype=torch.int64, device=device)
                    decoding_input_location_y = torch.zeros((1, ), dtype=torch.int64, device=device)
                else:
                    decoding_input_location_x = pred_x_array[pred_step-1].unsqueeze(0)
                    decoding_input_location_y = pred_y_array[pred_step-1].unsqueeze(0)

                decoding_input_embed = st_embedding(decoding_input_day_of_week, decoding_input_time_of_day, decoding_input_location_x, decoding_input_location_y)
                pred_x, pred_y, h = decoding(decoding_input_embed, h)

                pred_x_array[pred_step] = torch.argmax(pred_x, dim=-1) + 1
                pred_y_array[pred_step] = torch.argmax(pred_y, dim=-1) + 1

            generated = torch.stack((decoding_day, decoding_time_of_day, pred_x_array, pred_y_array), dim=-1).cpu().tolist()
            reference = torch.stack((decoding_day, decoding_time_of_day, label_location_x, label_location_y), dim=-1).cpu().tolist()
            result['generated'].append(generated)
            result['reference'].append(reference)

    with open(os.path.join(result_path, f'result.json'), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--st_embedding_pth', type=str, default='./checkpoint/task2/st_embedding.pth')
    parser.add_argument('--encoding_pth', type=str, default='./checkpoint/task2/encoding.pth')
    parser.add_argument('--decoding_pth', type=str, default='./checkpoint/task2/decoding.pth')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')

    task2(args.st_embedding_pth, args.encoding_pth, args.decoding_pth, device)
