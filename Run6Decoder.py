"""
    Experimental code for 6Decoder which is an effective IPv6 target generation algorithm.
"""

import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
import sys
import math
import numpy as np
from datetime import datetime
from TransformerDecoder import TransformerDecoder


# Parameters related to model training
SEED_FILE = 'data/Seed_S1_10K_32hex.txt'
MODEL_FILE  = 'data/model6decoder.pth'
CANDIDATES_FILE = 'data/candidates.txt'
MAX_LEN = 34    # <bos> + 32 nibbles + <eos>
BATCH_SIZE = 64
DATA_SHUFFLE = True
EPOCH_NUM = 5
LEARNING_RATE=5e-5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Model hyperparameters
N_LAYER = 6             # the number of Transformer Decoder layers
N_HEAD = 8              # the number of attention heads
D_FORWARD_DIM = 2048    # d_ff
D_MODEL = 512           # d_model
DROPOUT = 0.1           # dropout rate
TOP_K = 13               # select the top-k characters with the highest probabilities.



# Token-related variables
BOS, EOS = '<bos>', '<eos>'
tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', BOS, EOS]
token_to_id = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
               'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, BOS:16, EOS:17}
id_to_token = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
               10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f', 16:BOS, 17:EOS}
BOS_ID = token_to_id[BOS]
EOS_ID = token_to_id[EOS]
VOCAB_SIZE = len(tokens)



def token_encode(tokens):
    """ 
    Convert tokens to IDs
    nibble list -> <bos>ID + nibble ID list + <eos>ID 
    """
    token_ids = [BOS_ID]    # Start token ID
    # Traverse nibble list and convert each token into ID.
    for token in tokens:
        token_ids.append(token_to_id[token])
    token_ids.append(EOS_ID) # End token ID
    return token_ids



def token_decode(token_ids):
    """ 
    Convert IDs to tokens
    <bos>ID + nibble ID list + <eos>ID -> nibble list
    """
    tokens = []
    for idx in token_ids:
        # Skip start and end tokens
        if idx != BOS_ID and idx != EOS_ID:
            tokens.append(id_to_token[idx])
    return tokens    


class IPv6AddrSet(Dataset):
    """ Define custom IPv6 address dataset class """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    

def load_data(seed_file=SEED_FILE, batch_size=BATCH_SIZE):
    """ Load the IPv6 address dataset from seed file and return a DataLoader. """
    with open(seed_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # Encode the IPv6 address as a 32-length list of integers (0–15), 
    # with <bos> token ID at the beginning and <eos> token ID at the end.
    address = []
    for line in raw_data:
        address.append(token_encode(line.strip()))
    dataset = IPv6AddrSet(np.array(address))
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=DATA_SHUFFLE)
    return dataloader




class Model6Decoder(nn.Module):
    """
    Define 6Decoder model
    """
    def __init__(self, dict_size=VOCAB_SIZE, d_model=D_MODEL, nhead=N_HEAD,
                 d_ff=D_FORWARD_DIM, num_layers=N_LAYER, dropout=DROPOUT, 
                 activation=F.gelu):
        super(Model6Decoder, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=d_model)
        
        # Layer normalization layer
        norm = nn.LayerNorm(d_model)
        
        # An N-layer Transformer decoder stack
        self.decoder = TransformerDecoder(d_model=d_model, nhead=nhead, dropout=dropout,
                                          dim_feedforward=d_ff, num_layers=num_layers,
                                          norm=norm, activation=activation)
        
        # Linear output layer
        self.predictor = nn.Linear(d_model, dict_size)

    def forward(self, tgt, device=DEVICE):
        # Generate self-attention mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)

        # word embedding
        tgt = self.embedding(tgt)

        # permute(1, 0, 2) for reordering the tgt dimension to place the batch in the middle as batch_first is not enabled.
        out = self.decoder(tgt.permute(1, 0, 2), tgt_mask=tgt_mask)
        out = self.predictor(out)
        return out




def train_model(model, seed_file=SEED_FILE, model_file=None,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCH_NUM, 
                device=DEVICE):
    """ model training """
     
    dataloader = load_data(seed_file, batch_size) 

    # odel Loss Function and Optimizer 
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        data_progress = tqdm(dataloader, desc="Train...")
        for step, data in enumerate(data_progress, start=1):
            data = data.to(device)

            # Construct the training data and target data.
            tgt = data[:, :-1]
            tgt_y = data[:, 1:]
            
            # Perform the Transformer computation, and then pass the result to the final linear layer for prediction.
            out = model(tgt, device)
            loss = criteria(out.permute(1,2,0).contiguous(), tgt_y.to(dtype=torch.long))           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            total_loss += loss.item()

            # Update the training progress
            data_progress.set_description(f"Train... [epoch {epoch}/{epochs}, loss {(total_loss / step):.5f}]")

    # Save model parameters if needed
    if model_file is not None:
        torch.save(model.state_dict(), model_file)
    
    # Return the final average training loss.
    return total_loss / step




def ids_to_ipv6(addr):
    ''' Transform a list of 32 nibbles into an IPv6 address in colon-separated format. '''
    ipv6 = ''
    for i in range(len(addr)):
        ipv6 += id_to_token[addr[i]]
        if i%4 == 3 and i < 31:
            ipv6 += ':'
    return ipv6



def gen_addr_batch(model, top_k, head_num, head_batch, device=DEVICE):
    """ Generate one batch of IPv6 address """
    
    with torch.no_grad():
        # convert IPv6 address head nibble into a tensor.
        head_batch = torch.tensor(head_batch, dtype=torch.long, device=device)

        # Strip the last <eos> token.
        tgt = head_batch[:,:-1]

        i = 0
        while i < 32 - head_num:
            # model forward, out.shape=(sequence_len, batch_size, embed_dim)
            out = model(tgt, device)

             # 'out' contains the probability distribution over all tokens in the vocabulary. 
             # Exclude the last two tokens, which are <bos> and <eos>.
            _probas = out[-1, :, :-2]

            # Replace all values below top_k with -∞.
            indices_to_remove = _probas < torch.topk(_probas, top_k)[0][..., -1, None]
            _probas[indices_to_remove] = -float('Inf')

            # Apply the softmax operation so that tokens with higher probabilities are more likely to be selected.
            _probas = F.softmax(_probas, dim=-1)

            # Randomly select one token from the top-k based on their probabilities.
            y = torch.multinomial(_probas, num_samples=1)
            
            # Concatenate the selected token to the previously generated result.
            tgt = torch.cat((tgt, y), dim=-1)
            i += 1

        # Remove <bos> token and return generated addresses.
        ipv6list = list(map(ids_to_ipv6, tgt[:, 1:].tolist()))
        return ipv6list



def generate_target(model, top_k, budget, candidate_file, batch_size=BATCH_SIZE, device=DEVICE, head='2'):
    ''' Generate a certain number (budget) of IPv6 addresses and write them to a file. '''
    
    head_num = len(head)   # the length of address head nibble
    
    # encode address head nibble
    head_tokens_ids = token_encode(head)

    # Copy a single address head nibble into batch mode.
    head_batch = [head_tokens_ids for i in range(batch_size)]
    
    model.eval() # Switch the model to evaluation mode.

    # Generate IPv6 addresses in batches.
    addrs = set()
    progress_bar = tqdm(total=budget, desc="Generating...") # Display a progress bar
    while len(addrs) < budget:
        gen_addr = gen_addr_batch(model, top_k, head_num, head_batch, device=device)
        addrs.update(gen_addr)
        
        # Adjust the progress bar based on the number of IPv6 addresses to be created.
        progress_bar.n = len(addrs)
        progress_bar.refresh()
            
    # Append a newline character to the end of each IPv6 address.
    addrn = list(map(lambda s: s + "\n", addrs))
    
    # Write the generated addresses to a file.
    with open(candidate_file, 'w') as f:
        f.writelines(addrn[:budget])




if __name__ == '__main__':
    '''
    程序运行示例：
    python Run6Decoder.py --seed_file=data/Seed_S1_10K_32hex.txt \
                          --model_file=data/model6decoder.pth \
                          --candidate_file=data/candidates.txt \
                          --batch_size=64 \
                          --epochs=10 \
                          --learning_rate=5e-5 \
                          --top_k=13 \
                          --device=cuda:0 \
                          --budget=100000
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', action='store_true', default=False, help='no train flag')
    parser.add_argument('--seed_file', default=SEED_FILE, type=str, required=False, help='IPv6 seed set file for training')
    parser.add_argument('--model_file', default=MODEL_FILE, type=str, required=False, help='model parameters file')
    parser.add_argument('--candidate_file', default=CANDIDATES_FILE, type=str, required=False, help='generated candidates file')
    parser.add_argument('--n_layer', default=N_LAYER, type=int, required=False, help='the number of TransformerDecdoer layers')
    parser.add_argument('--n_head', default=N_HEAD, type=int, required=False, help='the number of self-attention heads')
    parser.add_argument('--d_ff', default=D_FORWARD_DIM, type=int, required=False, help='feed-forward dimension')
    parser.add_argument('--d_model', default=D_MODEL, type=int, required=False, help='model dimension')
    parser.add_argument('--dropout', default=DROPOUT, type=float, required=False, help='dropout rate')
    parser.add_argument('--top_k', default=TOP_K, type=int, required=False, help='select from top-k tokens with the highest probabilities')
    parser.add_argument('--epochs', default=EPOCH_NUM, type=int, required=False, help='training epochs')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, required=False, help='batch size during model training and evaluation')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, required=False, help='learning rate during training') 
    parser.add_argument('--budget', default=100000, type=int, required=False, help='the number candidate addresses to be generated')   
    parser.add_argument('--device', default=DEVICE, type=str, required=False, help='training and evaluation device')
    args = parser.parse_args()

    # Construct the model
    model = Model6Decoder(d_model=args.d_model, nhead=args.n_head, d_ff=args.d_ff, 
                            num_layers=args.n_layer, dropout=args.dropout).to(args.device)
    
    # Model training
    if args.no_train:
        model.load_state_dict(torch.load(args.model_file))  # load model parameters
    else:
        train_model(model=model, seed_file=args.seed_file, model_file=args.model_file,
                batch_size=args.batch_size, lr=args.learning_rate, epochs=args.epochs, 
                device=args.device)          

    # Generate candidate address
    generate_target(model=model, top_k=args.top_k, budget=args.budget, 
                    candidate_file=args.candidate_file, batch_size=2048, device=args.device)

