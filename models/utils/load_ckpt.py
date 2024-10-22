import torch

def load_and_freeze_model(model, checkpoint_path):
    """
    Loads a checkpoint into the model and freezes its parameters.

    Args:
    model (torch.nn.Module): The model to load the checkpoint into.
    checkpoint_path (str): Path to the checkpoint file.

    Returns:
    torch.nn.Module: The model with loaded weights and frozen parameters.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    sd=checkpoint['state_dict']
    re_sd={
                k.replace('model.', ''): v 
                for k, v in sd.items() 
                if 'rgbGAN' not in k
            }
    model.load_state_dict(re_sd)

    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model