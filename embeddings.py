import torch


def create_sine_positional_embedding(N:int,max_len:int,d_model:int):

    # Create positions and dimensions
    positions = torch.arange(max_len).unsqueeze(1)
    dims = torch.arange(0,d_model,2)

    # Calculate the denominator
    denominator = torch.exp(torch.log(positions) - math.log(N) * dims / d_model)

    # Create the positional embedding
    pos_embed = torch.empty(max_len,d_model)
    pos_embed[:,0::2] = torch.sin(denominator)
    pos_embed[:,1::2] = torch.cos(denominator)

    return pos_embed


