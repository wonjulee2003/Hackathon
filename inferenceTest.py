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
#import nvtx

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
    in_B = torch.from_numpy(B).float().view(-1, data_length, 1)
    in_F = torch.from_numpy(Freq).float().view(-1, 1)
    in_T = torch.from_numpy(Temp).float().view(-1, 1)
    in_D = torch.from_numpy(Hdc).float().view(-1, 1)
    out_H = torch.from_numpy(H).float().view(-1, data_length, 1)

    # Normalize
    in_B = (in_B-torch.mean(in_B))/torch.std(in_B)
    in_F = (in_F-torch.mean(in_F))/torch.std(in_F)
    in_T = (in_T-torch.mean(in_T))/torch.std(in_T)
    in_D = (in_D-torch.mean(in_D))/torch.std(in_D)

    MeanH = torch.std(out_H)
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

# Config the model testing

def main():

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 2000
    BATCH_SIZE = 128
    DECAY_EPOCH = 150
    DECAY_RATIO = 0.9
    LR_INI = 0.004

    # Select GPU as default device
    device = torch.device("cuda")

    # Load dataset
    dataset, normH = load_dataset()
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network
    net = Transformer(
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
      dim_feedforward_projecter=40).to(device)

    # Load trained parameters
    state_dict = torch.load('/scratch/gpfs/sw0123/Model_TransformerTest.sd')
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print("Model is loaded!")

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Test the network
    with torch.no_grad():
        for in_B, in_F, in_T, in_D, out_H, out_H_head in test_loader:

            # Create dummy out_H_head                            
            outputs = torch.zeros(out_H.size()).to(device)
            tgt = (torch.rand(out_H.size())*2-1).to(device)
            tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size())                        

            # Compute inference
            for t in range(1, out_H.size()[1]):   
                outputs = net(src=in_B.to(device),tgt=tgt.to(device),var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)                     
                tgt[:,t,:] = outputs[:,t-1,:]     
            outputs = net(in_B.to(device),tgt.to(device),torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)

            # Save results
            with open("/scratch/gpfs/sw0123/MagNet_Testing/pred.csv", "a") as f:
                np.savetxt(f, (outputs[:,:-1,:]*normH[1]+normH[0]).squeeze(2).cpu().numpy())
                f.close()
            with open("/scratch/gpfs/sw0123/MagNet_Testing/meas.csv", "a") as f:
                np.savetxt(f, (out_H[:,1:,:]*normH[1]+normH[0]).squeeze(2).cpu().numpy())
                f.close()
        print("Testing finished! Results are saved!")

if __name__ == "__main__":
    main()
