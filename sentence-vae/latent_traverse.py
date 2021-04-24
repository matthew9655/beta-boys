import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE\

def main(args):
    dataset= PTB(
            data_dir=args.data_dir,
            split='train',
            create_data=False,
            max_sequence_length=50,
            min_occ=1
    )
    data_loader = DataLoader(
                dataset=dataset,
                batch_size=5,
                shuffle=True,
                num_workers=4,
                pin_memory=True
    )

    # load model
    with open(args.data_dir + '/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']
    
    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )
    model.load_state_dict(torch.load(args.load_checkpoint))
    model.cuda()
    
    # first batch 
    first_batch = next(iter(data_loader))
    

    print('TRAVERSALS')
    num_samp = 1
    z = torch.randn((num_samp, args.latent_size)).cuda()
    num_latent_traverse = 10
    latent_traverse_arr = torch.linspace(-3, 3, 10)
    num_latent_dims = args.latent_size

        
    zs = torch.zeros(num_samp, num_latent_traverse * num_latent_dims, num_latent_dims)
    for i in range(num_samp):
        samp_z = z[i]
        for j in range(num_latent_dims):
            for k in range(num_latent_traverse):
                temp_samp_z = torch.clone(samp_z)
                temp_samp_z[j] = latent_traverse_arr[k]
                zs[i][(10 * j) + k] = temp_samp_z

    for i in range(num_samp):
        inter_cuda_z = to_var(zs[i]).float()
        samples,_ = model.inference(z=inter_cuda_z)
        print('sample {}'.format(i))
        print('---------------')
        print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        print('---------------')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='trump_data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=280)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=20)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
    