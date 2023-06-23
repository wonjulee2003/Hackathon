# Import necessary packages

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import time
#import nvtx

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
# from ffcv.fields.decoders import IntDecoder

# ffcv citation:
#@misc{leclerc2022ffcv,
    #author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    #title = {{FFCV}: Accelerating Training by Removing Data Bottlenecks},
    #year = {2022},
    #howpublished = {\url{https://github.com/libffcv/ffcv/}},
    #note = {commit xxxxxxx}
#}


# Define model structures and functions

class Transformer(nn.Module):
    def __init__(self, 
        input_size :int,
        dec_seq_len :int,
        max_seq_len :int,
        out_seq_len :int,
        dim_val :int,  
        n_encoder_layers :int,
        n_decoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_decoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        dim_feedforward_decoder :int,
        dim_feedforward_projecter :int,
        num_var: int=3
        ): 

        #   Args:
        #    input_size: int, number of input variables. 1 if univariate.
        #    dec_seq_len: int, the length of the input sequence fed to the decoder
        #    max_seq_len: int, length of the longest sequence the model will receive. Used in positional encoding. 
        #    out_seq_len: int, the length of the model's output (i.e. the target sequence length)
        #    dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_decoder_layers: int, number of stacked encoder layers in the decoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_decoder: float, the dropout rate of the decoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
        #    dim_feedforward_decoder: int, number of neurons in the linear layer of the decoder
        #    dim_feedforward_projecter :int, number of neurons in the linear layer of the projecter
        #    num_var: int, number of additional input variables of the projector

        super().__init__() 

        self.dec_seq_len = dec_seq_len
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.decoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, input_size))
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)
        self.projector = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_val))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
            )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor, var: Tensor, device) -> Tensor:

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)
        enc_seq_len = 128

        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat([src,var],dim=2))

        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        batch_size = src.size()[0]
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(device)
        output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=None
            ) 
        output= self.linear_mapping(output)

        return output

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz1: int, sz2: int) -> Tensor:
    #Generates an upper-triangular matrix of -inf, with zeros on diag.
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the dataset

def load_dataset(data_length=128):
    # Load .json Files
    with open('/scratch/gpfs/sw0123/Dataset_sine.json','r') as load_f:
        DATA = json.load(load_f)
    B = DATA['B_Field']
    B = np.array(B)
    Freq = DATA['Frequency']
    Freq = np.log10(Freq)  # logarithm, optional
    Temp = DATA['Temperature']
    Temp = np.array(Temp)      
    Hdc = DATA['Hdc']
    Hdc = np.array(Hdc)       
    H = DATA['H_Field']
    H = np.array(H)

    # Format data into tensors

    #to_tensor = ToTensor()
    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)
    #in_B = to_tensor(B).view(-1, data_length, 1)
    in_F = torch.from_numpy(Freq).float().view(-1, 1)
    #in_F = to_tensor(Freq).view(-1, data_length, 1)
    in_T = torch.from_numpy(Temp).float().view(-1, 1)
    #in_T = to_tensor(Temp).view(-1, data_length, 1)
    in_D = torch.from_numpy(Hdc).float().view(-1, 1)
    #in_D = to_tensor(Hdc).view(-1, data_length, 1)
    out_H = torch.from_numpy(H).float().view(-1, data_length, 1)
    #out_H = to_tensor(H).view(-1, data_length, 1)


    # Normalize
    in_B = (in_B-torch.mean(in_B))/torch.std(in_B)
    in_F = (in_F-torch.mean(in_F))/torch.std(in_F)
    in_T = (in_T-torch.mean(in_T))/torch.std(in_T)
    in_D = (in_D-torch.mean(in_D))/torch.std(in_D)

    MeanH = torch.mean(out_H)
    StdH = torch.std(out_H)
    out_H = (out_H-MeanH)/StdH

    # Save the normalization coefficients for reproducing the output sequences
    # For model deployment, all the coefficients need to be saved.
    normH = [MeanH,StdH]

    # Attach the starting token and add the noise
    head = 0.1 * torch.ones(out_H.size()[0],1,out_H.size()[2])
    out_H_head = torch.cat((head,out_H), dim=1)
    out_H = out_H_head
    out_H_head = out_H_head + (torch.rand(out_H_head.size())-0.5)*0.1

    print(in_B.size())
    print(in_F.size())
    print(in_T.size())
    print(in_D.size())
    print(out_H.size())
    print(out_H_head.size())

    return torch.utils.data.TensorDataset(in_B, in_F, in_T, in_D, out_H, out_H_head), normH

# Config the model training

def main():
    print("Main loop entered!")

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 100
    BATCH_SIZE = 1024
    DECAY_EPOCH = 150
    DECAY_RATIO = 0.9
    LR_INI = 0.004

    # Set # of epochs to discard due to warmup 
    DISCARD = 10

    # Select GPU as default device
    device = torch.device("cuda")

    to_device = ToDevice(device, non_blocking=True)

    # Record initial time
    start_time = time.time()

    #torch.cuda.nvtx.range_push("load_data")
    # Load dataset
    dataset, normH = load_dataset()

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    train_loader = Loader(train_dataset, batch_size=BATCH_SIZE, order=OrderOption.RANDOM, **kwargs)
    valid_loader = Loader(valid_dataset, batch_size=BATCH_SIZE, order=OrderOption.RANDOM, **kwargs)
    #torch.cuda.nvtx.range_pop()

    # Setup network
    net = to_device(Transformer(
      dim_val=24,
      input_size=1, 
      dec_seq_len=129,
      max_seq_len=129,
      out_seq_len=129, 
      n_decoder_layers=1,
      n_encoder_layers=1,
      n_heads=4,
      dropout_encoder=0.0, 
      dropout_decoder=0.0,
      dropout_pos_enc=0.0,
      dim_feedforward_encoder=40,
      dim_feedforward_decoder=40,
      dim_feedforward_projecter=40))

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Create list to store epoch times
    times=[]

    # Train the network
    for epoch_i in range(NUM_EPOCH):
        #torch.cuda.nvtx.range_push("epoch + " + str(epoch_i))

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        torch.cuda.synchronize()
        start_epoch = time.time()

        #torch.cuda.nvtx.range_push("train-loop")
        for in_B, in_F, in_T, in_D, out_H, out_H_head in train_loader:
            #torch.cuda.nvtx.range_push("zero gradient")
            for param in net.parameters():
                param.grad = None
            #torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("model_data_in")
            output = net(src=to_device(in_B), tgt=to_device(out_H_head), var=torch.cat((to_device(in_F), to_device(in_T), to_device(in_D)), dim=1))
            #torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("loss")
            loss = criterion(output[:,:-1,:], to_device(out_H)[:,1:,:])
            #torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("backward")
            loss.backward()
            #torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.25)
            #torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("optimizer")
            optimizer.step()
            #torch.cuda.nvtx.range_pop()

            epoch_train_loss += loss.item()

        #torch.cuda.nvtx.range_pop()

        #torch.cuda.nvtx.range_push("validation")
        # Compute validation
        with torch.no_grad():
            net.eval()
            epoch_valid_loss = 0
            for in_B, in_F, in_T, in_D, out_H, out_H_head in valid_loader:
                output = net(src=to_device(in_B), tgt=to_device(out_H_head), var=torch.cat((to_device(in_F), to_device(in_T), to_device(in_D)), dim=1))
                loss = criterion(output[:,:-1,:], to_device(out_H)[:,1:,:])
                epoch_valid_loss += loss.item()
        #torch.cuda.nvtx.range_pop()
        
        # Record epoch time 
        torch.cuda.synchronize()
        end_epoch = time.time()
        times.append(end_epoch-start_epoch)

        if (epoch_i+1)%200 == 0:
          print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
              f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
        
        #torch.cuda.nvtx.range_pop()


    elapsed = time.time() - start_time
    print(f"Total Time Elapsed: {elapsed}")    
    print(f"Average time per Epoch: {sum(times[DISCARD:])/NUM_EPOCH}")
    
    # Save the model parameters
    #torch.save(net.state_dict(), "/scratch/gpfs/sw0123/Model_TransformerTest.sd")
    print("Training finished! Model is saved!")

    timeNP = np.array(times)
    np.save("epochTime", timeNP)

if __name__ == "__main__":
    main()