import torch

def bottleneck(encoder_outs):
    # T x B x H
    # Normalize all, so partial things also work.
    # encoder_outs = torch.nn.functional.normalize(encoder_outs, dim=2, p=2)
    context, _ = torch.max(encoder_outs, dim=0)
    context = torch.nn.functional.normalize(context, dim=1, p=2)
    return context
