import torch

def move_to(sample, device):
    def _move(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move(x) for x in maybe_tensor]
        else:
            return maybe_tensor
    return _move(sample)

