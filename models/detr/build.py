# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
from extensions.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from .transformer import (TransformerEncoder, TransformerEncoderLayer)


def build_preencoder(num_group, group_size, dim):
    return PointnetSAModuleVotes(
        radius=0.2,
        nsample=group_size,
        npoint=num_group,
        mlp=[0, 64, 128, dim],
        normalize_xyz=True,
    )


def build_encoder(ndim, nhead, nlayers=3, ffn_dim=128, dropout=0.1):
    encoder_layer = TransformerEncoderLayer(
        d_model=ndim,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="relu",
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=nlayers, norm=nn.LayerNorm(ndim)
    )
    return encoder
