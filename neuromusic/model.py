import os
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from datetime import datetime
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple

INPUT_SPACE = 130

def_hparams_dict = {
    'batch_size': 256,
    'lr': 0.001,
    'lr_decay': 0.9999,
    'z_size': 256,
    'enc_size': 512,
    'cond_size':128,
    'dec_size': 512,
    'dropout_rate': 0.2
}
HParams = namedtuple('HParams', sorted(def_hparams_dict))
def_hparams = HParams(**def_hparams_dict)
# HParams = namedtuple('HParams', [ 'batch_size', 'lr', 'lr_decay' 'z_size', 
#                                  'enc_size', 'cond_size', 'dec_size', 'dropout_rate' ])
# def_hparams = HParams(
#     256,#batch_size
#     0.001,# lr=1e-3,
#     0.9999,# lr_decay=0.9999,
#     256,# z_size=256,
#     512,# enc_size=512,
#     128,# cond_size=128,
#     512,# dec_size=512,
#     0.2# dropout_rate=0.2
# )

class MusicVAE(nn.Module):
    def __init__(self, hparams: HParams):
        super(MusicVAE, self).__init__()
        self._hparams = hparams
        self.batch = 1
        
        self.encoderLSTM = nn.LSTM(
            input_size=INPUT_SPACE,
            hidden_size=hparams.enc_size,
            bidirectional=True,
            batch_first=True
        )
        self.mu = nn.Linear(
            in_features=hparams.enc_size*2, # Bc it's bidiriectional
            out_features=hparams.z_size
        )
        self.logvar = nn.Linear(
            in_features=hparams.enc_size*2, # Bc it's bidiriectional
            out_features=hparams.z_size
        )
        
        self.dropout = nn.Dropout(p=hparams.dropout_rate)
        
        self.z_linear = nn.Linear(
            in_features=hparams.z_size,
            out_features=hparams.cond_size
        )

        self.conductorLSTM = nn.LSTMCell(
            input_size=hparams.cond_size,
            hidden_size=hparams.cond_size
        )
        self.c_linear = nn.Linear(
            in_features=hparams.cond_size,
            out_features=hparams.dec_size
        )
        self.decoderLSTM = nn.LSTMCell(
            input_size=INPUT_SPACE+hparams.dec_size, # pass the last output from the prevous group
            hidden_size=INPUT_SPACE
        )

    def _encode(self, x):
        # x: (batch, seq_len*16, INPUT_SPACE)
        z_seq, _ = self.encoderLSTM(x.view(self.batch, -1, INPUT_SPACE)) # z_seq: (batch, seq_len*16, enc_size*2)
        z1 = z_seq[:, -1] # z1: (batch, enc_size*2)
        z1 = self.dropout(z1)
        return F.relu(self.mu(z1)), F.relu(self.logvar(z1)) # (batch, z_size)
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # eps = torch.zeros_like(std)
        return mu + eps*std
    
    def _decode_single_bar(self, last_note, cond_initial, cond_hidden, cond_cell):
        # last_note: (batch, INPUT_SPACE)
        # cond_initial: (batch, cond_size)
        # cond_hidden: (batch, cond_size)
        batch = cond_initial.shape[0]
        
        cond_out, cond_cell = self.conductorLSTM(cond_initial, (cond_hidden, cond_cell)) # cond_out: (batch, cond_size)
        dec_init = F.relu(self.c_linear(self.dropout(cond_out))) # dec_init: (batch, dec_size)
        
        notes = []
        dec_h, dec_c = torch.zeros(batch, INPUT_SPACE), torch.zeros(batch, INPUT_SPACE)
        for i in range(16):
            dec_in = torch.cat((last_note, dec_init), dim=1) # dec_in: (batch, INPUT_SPACE+dec_size)
            dec_h, dec_c = self.decoderLSTM(dec_in, (dec_h, dec_c)) # dec_h: (batch, INPUT_SPACE)
            last_note = F.softmax(dec_h, dim=1) # note: (batch, INPUT_SPACE)
            notes.append(last_note)

        # notes_tensor = torch.cat(notes).view(notes[0].shape[0], 16, -1) # notes: (batch, 16, INPUT_SPACE)
        return notes, cond_out, cond_cell
            
    
    def _decode(self, z, seq_len=1, return_tensor=True, return_cond_state=False):
        # z: (batch, z_size)
        batch = z.shape[0]
        cond_size = self._hparams.cond_size
        
        cond_init = F.relu(self.z_linear(self.dropout(z))) # z2: (batch, cond_size)
        cond_h, cond_c = torch.zeros(batch, cond_size), torch.zeros(batch, cond_size) # cond_h: (batch, cond_size)
        last_note = torch.zeros(batch, INPUT_SPACE) # last_note: (batch, INPUT_SPACE)
        notes = []
        for i in range(seq_len):
            new_notes, cond_h, cond_c = self._decode_single_bar(last_note, cond_init, cond_h, cond_c) # new_notes [(batch, INPUT_SPACE)]
            last_note = new_notes[-1]
            notes += new_notes
        
        return_val = notes
        if return_tensor:
            return_val = torch.cat(notes, 1).view(batch, -1, INPUT_SPACE) # notes: (batch, seq_len*16, INPUT_SPACE)
        if return_cond_state:
            return_val = (return_val, (cond_h, cond_c))
        return return_val
        
    def forward(self, input):
        # input: (batch, seq_len*16, INPUT_SPACE)
        batch = input.shape[0]
        seq_len = input.shape[1]//16
        
        # mu, logvar = torch.randn((1, def_hparams.z_size)), torch.randn((1, def_hparams.z_size))
        mu, logvar = self._encode(input) # mu, logvar: (batch, z_size)
        
        z = self._reparameterize(mu, logvar) # z: (batch, z_size)
        
        out = self._decode(z, seq_len) # out: (batch, seq_len*16, INPUT_SPACE)
        return out, (mu, logvar)

class VAETrainer():
    def __init__(self, hparams: HParams, train_loader, test_loader, log_base_dir='runs'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hparams = hparams
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model = MusicVAE(hparams).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, hparams.lr_decay)
        
        base_dir = os.path.join(log_base_dir, f"musicvae-{datetime.now().strftime('%m-%d_%H:%M:%S')}")
        
        log_dir = os.path.join(base_dir, 'logs')
        self.writer = SummaryWriter(log_dir)
        
        self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        
    def _loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def test(self):
        self.model.eval()
        test_loss = 0
        batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                x = batch.to(self.device)
                out, (mu, logvar) = self.model(x)
                loss = self._loss_function(out, x, mu, logvar)
                
                test_loss += loss
                batches += 1
        return test_loss / batches
    
    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        batches = 0
        for batch_idx, batch in enumerate(self.train_loader):
            
            x = batch.to(self.device)
            self.optim.zero_grad()
            
            print('running model')
            out, (mu, logvar) = self.model(x)
            loss = self._loss_function(out, x, mu, logvar)
            
            print('backward')
            with torch.autograd.profiler.profile() as prof2:
                loss.backward()
            print(prof2.key_averages().table(sort_by="self_cpu_time_total"))

            return
            # self.optim.step()
            # train_loss += loss.item()
            # batches += 1
        # return train_loss / batches
    
    def train(self, n_epochs, checkpoint_freq=10):
        hp_dict = dict(self.hparams._asdict())
        for epoch in range(n_epochs):
            train_loss = self._train_epoch()
            self.writer.add_scalar('train_loss', train_loss)
            
            test_loss = self.test()
            self.writer.add_scalar('test_loss', test_loss)
            
            self.scheduler.step(epoch)
            
            
            if epoch == 0:
                self.writer.add_graph(self.model)
            
            if epoch % checkpoint_freq == 0:
                self.writer.add_hparams(hp_dict, {'hparam/loss': test_loss})
                path = os.path.join(self.checkpoint_dir, f'epoch-{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optim.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                })
            
