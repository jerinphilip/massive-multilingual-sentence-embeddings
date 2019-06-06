from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    add_args(parser)
    return parser


def add_args(parser):
    parser.add_argument('--dict_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--source_lang', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--target_lang', type=str, required=True)
    parser.add_argument('--encoder_embedding_dim', type=int, default=512)
    parser.add_argument('--encoder_hidden_size', type=int, default=512)
    parser.add_argument('--encoder_num_layers', type=int, default=3)
    parser.add_argument('--encoder_bidirectional', type=bool, default=True)
    parser.add_argument('--decoder_embedding_dim', type=int, default=512)

    # Set this to 2 * encoder
    parser.add_argument('--decoder_hidden_size', type=int, default=1024)
    parser.add_argument('--decoder_num_layers', type=int, default=3)
    parser.add_argument('--decoder_bidirectional', type=bool, default=False)
    parser.add_argument('--decoder_output_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--clip_grad_norm', type=float, default=5)
    parser.add_argument('--update_every', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=5)


    # Distributed arguments
    parser.add_argument('--distributed_rank', type=int, required=False)
    parser.add_argument('--distributed_backend', type=str, default='nccl')
    parser.add_argument('--distributed_world_size', type=int, required=True)
    parser.add_argument('--distributed_master_addr', type=str, required=True)
    parser.add_argument('--distributed_port', type=int, required=True)
    parser.add_argument('--device', type=int, required=False)

    # 
    parser.add_argument('--progress', type=str, choices=['tqdm', 'none'], default='tqdm')
    parser.add_argument('--worker_output', action='store_true')
