import torch
import torch.nn as nn

from models.vit_lr.TransformerBlock import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        num_heads,
        tgt_len,
        ff_dim,
        dropout,
        latent_replay_block,
        device,
    ):
        super().__init__()

        # Set latent replay block
        assert (
            -1 <= latent_replay_block < num_blocks
        ), "Invalid latent replay block selection."
        self.latent_replay_block = latent_replay_block
        self.block_id = [0] * num_blocks

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    tgt_len=tgt_len,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.current_object = -1

    def forward(self, x, get_activation=False):
        is_pattern, x = x

        if is_pattern:
            blocks_to_run = self.blocks
        else:
            assert self.latent_replay_block > -1, "Latent replay block not set."
            blocks_to_run = self.blocks[self.latent_replay_block :]

        activation = None
        for i, block in enumerate(blocks_to_run):
            if get_activation and i == self.latent_replay_block:
                activation = x.clone().detach()
            x = block(x)

            torch.save(
                x,
                f"activations/{i}/transformer_block_{self.current_object}_{self.block_id[i]}.pth",
            )
            self.block_id[i] += 1
            print(self.current_object)

        if get_activation:
            return x, activation
        else:
            return x
