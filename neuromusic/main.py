from data import get_data
from model import VAETrainer, def_hparams, MusicVAE

import torch
import torch.nn.functional as F

INPUT_SPACE = 130

DATA = '/storage/piano_data.npz'

if __name__ == "__main__":
    train_loader, test_loader = get_data(DATA)
    trainer = VAETrainer(def_hparams, train_loader, test_loader)
    trainer.train(100)
    # trainer._train_epoch()
    # from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/musicvae2/logs')
    # seq_len = 4
    # dummy_input = torch.randn(1, seq_len*16, INPUT_SPACE).to(trainer.device)
    # print('graph model')
    # trainer.model.eval()
    # writer.add_graph(trainer.model, dummy_input)
    # writer.export_scalars_to_json("./all_scalars.json")
    # writer.close()
    # # trainer.train(10)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print('Create model...')
    # model = MusicVAE(def_hparams).to(device)


    # # last_note=torch.randint(0, 1, (1, INPUT_SPACE), dtype=torch.float)
    # # cond_init = torch.ones((1, def_hparams.cond_size), dtype=torch.float)
    # # cond_hidden = torch.randn((1, def_hparams.cond_size))
    # # cond_cell = torch.randn((1, def_hparams.cond_size))
    # # target_notes=torch.randint(0, 1, (1, 16 ,INPUT_SPACE), dtype=torch.float)

    # seq_len = 4
    # z = torch.randn((1, def_hparams.z_size))
    # target_notes=torch.randint(0, 1, (1, 16*seq_len ,INPUT_SPACE), dtype=torch.float)

    # print('Start test...')
    # with torch.autograd.profiler.profile() as prof1:
    #     # out_notes, _, _ = model._decode_single_bar(last_note, cond_init, cond_hidden, cond_cell)
    #     notes = model._decode(z, seq_len)
    # print(prof1.key_averages().table(sort_by="self_cpu_time_total"))

    # print()
    # # notes = torch.cat(out_notes, 1).view(1, -1, INPUT_SPACE)
    # loss = F.binary_cross_entropy(notes, target_notes)
    # with torch.autograd.profiler.profile() as prof2:
    #     loss.backward()
    # print(prof2.key_averages().table(sort_by="self_cpu_time_total"))
