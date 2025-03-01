# Sheng Wang at Feb 22 2023
# Based on LoRA-ViT: https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# Modified by Fares Abawi (@fabawi).

import logging
import os
import math
from typing import Optional, List, Dict, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

from models.transformer import SimpleTransformer


class _LoRALayerHead(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x) + self.w_b(self.w_a(x))


class LoRA_Head(nn.Module):
    """Applies low-rank adaptation to a head (e.g., classification head) of a model.

    Args:
        head_model: The head model (e.g., a linear layer).
        rank: Rank of LoRA.
    """

    def __init__(self, head_model: nn.Module, rank: int):
        super(LoRA_Head, self).__init__()
        assert rank > 0
        self.head_model = head_model
        self.rank = rank

        # Freeze the original head parameters
        for param in head_model.parameters():
            param.requires_grad = False

        # Create LoRA layers
        self.w_a = nn.Linear(head_model.in_features, rank, bias=False)
        self.w_b = nn.Linear(rank, head_model.out_features, bias=False)

        if self.training:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_model(x) + self.w_b(self.w_a(x))

    def save_lora_parameters(self, filename: str) -> None:
        """Saves LoRA parameters to a file."""
        assert filename.endswith(".safetensors")
        lora_params = {
            "w_a": self.w_a.weight,
            "w_b": self.w_b.weight,
        }
        save_file(lora_params, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """Loads LoRA parameters from a file."""
        assert filename.endswith(".safetensors")
        with safe_open(filename, framework="pt") as f:
            self.w_a.weight = Parameter(f.get_tensor("w_a"))
            self.w_b.weight = Parameter(f.get_tensor("w_b"))



def save_lora_heads(
    lora_heads: Dict[str, Union[nn.Sequential, nn.Module]], 
    checkpoint_dir: str = "./.checkpoints/lora", 
    postfix: str = "_last", 
    extension: str = "safetensors"
):
    """Saves LoRA parameters for heads that contain a LoRA_Head layer."""
    for head_name, head in lora_heads.items():
        try:
            if hasattr(head, "__iter__"):  # Check if the head is iterable (e.g., nn.Sequential)
                # Iterate through the layers of the Sequential head
                for layer in head:
                    if isinstance(layer, LoRA_Head):
                        # Save LoRA parameters for this head
                        layer.save_lora_parameters(os.path.join(checkpoint_dir, f"lora-head-{head_name}{postfix}.{extension}"))
                        logging.info(f"Saved LoRA parameters for head {head_name} to {checkpoint_dir}.")
                        break  # Stop after saving the first LoRA_Head layer
            elif isinstance(head, LoRA_Head):
                # Directly save LoRA parameters if the head is a LoRA_Head
                head.save_lora_parameters(os.path.join(checkpoint_dir, f"lora-head-{head_name}{postfix}.{extension}"))
                logging.info(f"Saved LoRA parameters for head {head_name} to {checkpoint_dir}.")
            else:
                # Skip heads that are not Sequential or LoRA_Head
                logging.warning(f"Head {head_name} is not a Sequential or LoRA_Head. Skipping.")
        except FileNotFoundError:
            logging.warning(f"Could not save LoRA parameters for head {head_name} to {checkpoint_dir}.")


def load_lora_heads(
    lora_heads: Dict[str, Union[nn.Sequential, nn.Module]], 
    checkpoint_dir: str = "./.checkpoints/lora", 
    postfix: str = "_last", 
    extension: str = "safetensors"
):
    """Loads LoRA parameters for heads that contain a LoRA_Head layer."""
    for head_name, head in lora_heads.items():
        try:
            if isinstance(head, nn.Sequential):
                # Iterate through the layers of the Sequential head
                for layer in head:
                    if isinstance(layer, LoRA_Head):
                        # Load LoRA parameters for this head
                        layer.load_lora_parameters(os.path.join(checkpoint_dir, f"lora-head-{head_name}{postfix}.{extension}"))
                        logging.info(f"Loaded LoRA parameters for head {head_name} from {checkpoint_dir}.")
                        break  # Stop after loading the first LoRA_Head layer
            elif isinstance(head, LoRA_Head):
                # Directly load LoRA parameters if the head is a LoRA_Head
                head.load_lora_parameters(os.path.join(checkpoint_dir, f"lora-head-{head_name}{postfix}.{extension}"))
                logging.info(f"Loaded LoRA parameters for head {head_name} from {checkpoint_dir}.")
            else:
                # Skip heads that are not Sequential or LoRA_Head
                logging.warning(f"Head {head_name} is not a Sequential or LoRA_Head. Skipping.")
        except FileNotFoundError:
            logging.warning(f"Could not find LoRA parameters for head {head_name} in {checkpoint_dir}.")
            logging.warning("If you are training the sub-model from scratch, this is expected.")
            logging.warning("If you are loading parts of a pre-trained model, this is expected for some heads.")


def apply_lora_to_sequential_head(sequential_head: nn.Sequential, rank: int) -> nn.Sequential:
    """Applies LoRA to specific layers (e.g., nn.Linear) within an nn.Sequential head.

    Args:
        sequential_head: The head as an nn.Sequential module.
        rank: Rank of LoRA.

    Returns:
        The modified nn.Sequential module with LoRA applied.
    """
    new_layers = []
    for layer in sequential_head:
        if isinstance(layer, nn.Linear):
            # Apply LoRA to this linear layer
            lora_layer = LoRA_Head(layer, rank)
            new_layers.append(lora_layer)
        else:
            # Keep other layers unchanged
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def apply_lora_heads(modality_heads: Dict[str, nn.Sequential], rank: int) -> Dict[str, nn.Sequential]:
    """Applies LoRA to a dictionary of modality heads.

    Args:
        modality_heads: Dictionary of modality heads (nn.Sequential).
        rank: Rank of LoRA.

    Returns:
        A dictionary of modality heads with LoRA applied.
    """
    return {modality_name: apply_lora_to_sequential_head(head, rank) for modality_name, head in modality_heads.items()}




















def apply_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer], rank: int,
                               layer_idxs: Optional[Dict[SimpleNamespace, List[int]]] = None,
                               modality_names: List[SimpleNamespace] = None):
    if modality_names is None:
        modality_names = list(modality_trunks.keys())
    if layer_idxs is None:
        layer_idxs = {}
    return nn.ModuleDict({modality_name: LoRA_SimpleTransformer(modality_trunk, rank, layer_idxs.get(modality_name, None)) for
                          modality_name, modality_trunk in modality_trunks.items() if modality_name in modality_names})


def save_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer],
                              checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):
    for modality_name, modality_trunk in modality_trunks.items():
        try:
            if isinstance(modality_trunk, LoRA_SimpleTransformer):
                modality_trunk.save_lora_parameters(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                logging.info(f"Saved LoRA parameters for modality {modality_name} to {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not save LoRA parameters for modality {modality_name} to {checkpoint_dir}.")


def load_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer],
                              checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):

    for modality_name, modality_trunk in modality_trunks.items():
        try:
            if isinstance(modality_trunk, LoRA_SimpleTransformer):
                modality_trunk.load_lora_parameters(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                logging.info(f"Loaded LoRA parameters for modality {modality_name} from {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not find LoRA parameters for modality {modality_name} in {checkpoint_dir}.")
            logging.warning("If you are training the sub-model from scratch, this is expected.")
            logging.warning("If you are loading parts of a pre-trained model, this is expected for some modalities.")


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, **kwargs):

        x = self.w(x, attn_mask=attn_mask) + self.w_b(self.w_a(x))
        return x


class LoRA_SimpleTransformer(nn.Module):
    """Applies low-rank adaptation to simple transformer with pytorch multihead attention.

    Args:
        transformer_model: a vision transformer model, see base_vit.py
        rank: rank of LoRA
        lora_layer_idxs: which layer we apply LoRA.

    Examples::
        >>> model = SimpleTransformer()
        >>> lora_model = LoRA_SimpleTransformer(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, transformer_model: SimpleTransformer, rank: int, lora_layer_idxs: Optional[List[int]] = None):
        super(LoRA_SimpleTransformer, self).__init__()

        assert rank > 0
        self.base_dim = transformer_model.blocks[0].attn.in_proj_bias.size()[0]//3
        dim = self.base_dim
        if lora_layer_idxs is not None:
            self.lora_layer_idxs = lora_layer_idxs
        else:
            self.lora_layer_idxs = list(range(len(transformer_model.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in transformer_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As.append(w_a_linear_qkv)
            self.w_Bs.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)

        if self.training:
            self.reset_parameters()
        self.lora_model = transformer_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, tokens: torch.Tensor, **kwargs) -> Tensor:
        return self.lora_model(tokens)


