import os
import sys
from tqdm import tqdm, trange
import torch
from .parser import create_parser
from .task import JointSpaceLearningTask

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    task = JointSpaceLearningTask(args)
    task.load_dataset()
    loaders = task.get_loader()
    device = torch.device('cuda:1') if torch.cuda.is_available() \
            else torch.device("cpu")

    task.build_trainer()
    task.trainer.to(device)
    for epoch in trange(args.num_epochs, leave=True):
        loss_sum = 0
        for dataset_idx, loader in enumerate(loaders):
            pbar = tqdm(enumerate(iter(loader)), total=len(loader), ascii='#', leave=True)
            for batch_idx, batch in pbar:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                        # print(key, batch[key].size())
                loss = task.trainer.run_update(batch)
                loss_sum += loss
                if batch_idx % args.update_every == 0:
                    # print(loss/batch['seq_length'])
                    state_dict = {
                        "epoch": epoch,
                        #"batch_idx": batch_idx,
                        "dataset_idx": dataset_idx,
                        "lpb": loss_sum/(batch_idx+1),
                        "lpt": loss_sum/((batch_idx+1)*batch["tgt_num_tokens"]),
                        "toks": batch["src_num_tokens"] + batch["tgt_num_tokens"],
                        # "src_toks": batch["src_num_tokens"],
                        # "tgt_toks": batch["tgt_num_tokens"]
                    }
                    pbar.set_postfix(**state_dict)

