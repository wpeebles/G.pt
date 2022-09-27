"""
This file contains the G.pt model and its building blocks (minGPT without masking, etc.).
"""
import math
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Modified from: https://github.com/karpathy/minGPT
    A version of minGPT without masking (we use diffusion instead).
    """

    def __init__(
        self, input_parameter_sizes, output_parameter_sizes, input_parameter_names,
        n_layer=12, n_head=12, n_embd=768, encoder_depth=1, decoder_depth=1, attn_pdrop=0.0,
        resid_pdrop=0.0, embd_pdrop=0.0, chunk_size=None, split_policy='chunk'
    ):
        # parameter_sizes is a list of integers indicating how many parameters are in each layer
        super().__init__()

        # Determine how many parameters are placed into each individual Transformer token:
        self.input_splits = self.build_splits(input_parameter_sizes, split_policy, chunk_size)
        self.output_splits = self.build_splits(output_parameter_sizes, split_policy, chunk_size)
        print(f'Using following input parameter splits: {self.input_splits}')
        block_size = len(self.input_splits)

        if split_policy == 'layer_by_layer':
            assert len(input_parameter_names) == block_size
        else:
            input_parameter_names = ['null'] * block_size

        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )

        self.block_size = block_size

        # Per-token encoder layers:
        self.input_parameter_projections = self.build_encoder(n_embd, encoder_depth, self.input_splits)
        self.ln_in = nn.LayerNorm(n_embd)

        # Per-token decoder layers:
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_parameter_projections = self.build_decoder(n_embd, decoder_depth, self.output_splits)

        self.num_output_heads = len(self.output_splits)
        self.apply(self._init_weights)

        print(f"number of parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def build_encoder(n_embd, encoder_depth, input_splits):
        # Create a unique MLP encoder for each token
        input_parameter_projections = nn.ModuleList()
        for param_chunk_size in input_splits:
            in_proj = [nn.Linear(param_chunk_size, n_embd, bias=False)]
            for _ in range(encoder_depth - 1):
                in_proj.append(nn.GELU())
                in_proj.append(nn.Linear(n_embd, n_embd, bias=False))
            in_proj = nn.Sequential(*in_proj)
            input_parameter_projections.append(in_proj)
        return input_parameter_projections

    @staticmethod
    def build_decoder(n_embd, decoder_depth, output_splits):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()
        for output_chunk_size in output_splits:
            out_proj = []
            for _ in range(decoder_depth - 1):
                out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
                out_proj.append(nn.GELU())
            out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)
        return output_parameter_projections

    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def configure_optimizers(nn_module, lr, wd, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm, torch.nn.Embedding, FrequencyEmbedder
        )
        for mn, m in nn_module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('pos_emb') or pn.endswith('cfg_loss_embedding') or pn.endswith("hypernet_z_tokens"):
                    # special case the position embedding parameter
                    # in the root GPT module as not decayed
                    no_decay.add(fpn)
        # decay.add('decoder._fsdp_wrapped_module.flat_param')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in nn_module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def encode_parameters(self, parameters):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        """
        assert parameters.dim() == 2
        split_parameters = torch.split(parameters, self.input_splits, dim=1)
        representations = []
        for parameter, in_proj in zip(split_parameters, self.input_parameter_projections):
            representations.append(in_proj(parameter))
        representations = torch.stack(representations, dim=1)  # (b, t, d)
        representations = self.ln_in(representations)
        assert representations.dim() == 3
        return representations

    def decode_parameters(self, features):
        """
        Apply a per-chunk decoding (only to the tokens corresponding to the noised/updated parameter vector),
        and concatenate them into a flattened parameter vector.
        """
        assert features.dim() == 3  # (b, t, d)
        output = []
        for t in range(self.num_output_heads):
            out_proj = self.output_parameter_projections[t]
            output.append(out_proj(features[:, t, :]))
        output = torch.cat(output, 1)  # (b, c)
        assert output.dim() == 2
        return output

    @staticmethod
    def flatten_parameter_sizes(parameter_sizes):
        assert len(parameter_sizes) > 0
        if not isinstance(parameter_sizes[0], (list, tuple)):  # list is already flat
            return parameter_sizes
        return [p for group in parameter_sizes for p in group]

    @staticmethod
    def build_splits(parameter_sizes, split_policy='chunk', chunk_size=None):
        """
        Determines how to split the input parameter vector into individual tokens.

        'chunk': Basic approach (naively concatenate all inputs together and chunk them
                 indiscriminately, each token can contain values from different layers and inputs)
        'layer_by_layer': each layer's parameters are contained in a SINGLE token,
                          no mixing across layers or inputs
        'chunk_within_layer': each layer's parameters are subdivided into MANY tokens,
                              no mixing across layers or inputs
        'chunk_within_input': each input's elements are subdivided into MANY tokens,
                              no mixing across inputs
        """
        if split_policy == 'chunk':
            # Chunk the parameter vector, not caring if one chunk contains parameters
            # from different layers:
            assert chunk_size is not None
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            total_n_params = sum(parameter_sizes)
            num = total_n_params // chunk_size
            splits = [chunk_size] * num
            remainder = total_n_params % chunk_size
            if remainder > 0:
                splits.append(remainder)
        elif split_policy == 'layer_by_layer':
            # Each layer's parameters belong to its own chunk:
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            splits = parameter_sizes
        elif split_policy == 'chunk_within_layer':
            # Chunk the parameter vector, ensuring that each chunk contains parameters
            # from a single layer only:
            assert chunk_size is not None
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            splits = []
            for param_size in parameter_sizes:
                num = param_size // chunk_size
                splits.extend([chunk_size] * num)
                remainder = param_size % chunk_size
                if remainder > 0:
                    splits.append(remainder)
        elif split_policy == 'chunk_within_input':
            splits = []
            for parameter_group in parameter_sizes:
                assert isinstance(parameter_group, (list, tuple))
                splits.extend(GPT.build_splits(parameter_group, 'chunk', chunk_size))
            return splits
        else:
            raise NotImplementedError
        return splits

    def forward(self, x):
        embeddings = self.encode_parameters(x)
        b, t, d = embeddings.size()
        assert t == self.block_size, f"Expected {self.block_size} tokens on dim=1, but got {t}"

        # forward the GPT model
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.decode_parameters(x)

        return x


class FrequencyEmbedder(nn.Module):

    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1).to('cuda', torch.float)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)  # (N, D * 2 * num_frequencies + D)
        return embedded


class Gpt(nn.Module):

    """
    The G.pt model.
    """

    def __init__(
        self,
        parameter_sizes,                    # A list of integers indicating the total number of parameters in each layer
        parameter_names,                    # A list of strings indicating the name of each layer in the input networks
        num_frequencies=128,                # number of frequencies sampled for embedding scalars
        max_freq_log2=20,                   # max log2 frequency for embedding scalars
        predict_xstart=True,                # if True, G.pt predicts signal (False = predict noise)
        absolute_loss_conditioning=False,   # if True, adds two extra input tokens indicating starting/target metrics
        **gpt_kwargs                        # Arguments for the Transformer model (depth, heads, etc.)
    ):
        super().__init__()
        self.predict_xstart = predict_xstart
        self.absolute_loss_conditioning = absolute_loss_conditioning

        input_parameter_sizes, output_parameter_sizes, input_parameter_names = \
            self.compute_token_sizes(parameter_sizes, parameter_names, num_frequencies)

        self.decoder = GPT(input_parameter_sizes, output_parameter_sizes, input_parameter_names, **gpt_kwargs)
        self.scalar_embedder = FrequencyEmbedder(num_frequencies, max_freq_log2)

        # Initialize with identity output:
        self.use_global_residual = True
        if self.use_global_residual:
            for out_proj in self.decoder.output_parameter_projections:
                out_proj[-1].weight.data.zero_()

    @staticmethod
    def get_scalar_token_size(num_frequencies):
        """
        Computes the size of each metadata token after being projected by the frequency embedder.
        """
        return num_frequencies * 2 + 1

    def compute_token_sizes(self, parameter_sizes, parameter_names, num_frequencies):
        """
        This function returns a few different lists which are used to construct the GPT model.

        input_parameter_sizes: A list that breaks-down the sizes of the different input vectors.
        output_parameter_sizes: A list that breaks-down the sizes of the different vectors the G.pt model will output.
        input_parameter_names: A list that contains string names for every individual input layer and scalar.

        For example, say we have a linear network with a (10, 784)-shape weight and a (10,)-shape bias, and we embed
        each input scalar into a 257-dimensional vector. Then this function might return the following:

        input_parameter_sizes: [[7840, 10], [7840, 10], [257], [257], [257], [257]]
        output_parameter_sizes: [[7840, 10]]  # G.pt only outputs denoised parameters
        input_parameter_names: ['weight', 'bias', 'weight', 'bias', 'timestep_embedding',
                               'loss_delta_embedding', 'target_loss_embedding', 'current_loss_embedding']

        These lists are used by the GPT class above to determine how to split the input vector into different tokens.
        """
        input_parameter_sizes = [deepcopy(parameter_sizes)]
        output_parameter_sizes = [deepcopy(parameter_sizes)]
        input_parameter_names = deepcopy(parameter_names)
        # account for the second weight vector that will be input:
        input_parameter_sizes.append(input_parameter_sizes[0])
        input_parameter_names.extend(input_parameter_names)
        # Account for the scalar inputs (diffusion timestep and loss/error/return inputs):
        scalar_token_size = [self.get_scalar_token_size(num_frequencies)]
        input_parameter_sizes.extend([scalar_token_size, scalar_token_size])
        input_parameter_names.extend(['timestep_embedding', 'loss_delta_embedding'])
        if self.absolute_loss_conditioning:
            input_parameter_sizes.extend([scalar_token_size, scalar_token_size])
            input_parameter_names.extend(['target_loss_embedding', 'current_loss_embedding'])
        return input_parameter_sizes, output_parameter_sizes, input_parameter_names

    def configure_optimizers(self, lr, wd, betas):
        """
        Sets up the AdamW optimizer for G.pt (no weight decay on the positional embeddings or layer norm biases).
        """
        return GPT.configure_optimizers(self, lr, wd, betas)

    @torch.no_grad()
    def gradient_norm(self):
        """
        Computes the gradient norm for monitoring purposes.
        """
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def embed_loss(self, loss_target, loss_prev):
        """
        Embeds the input loss/error/return scalars into a frequency-based vector representation
        """
        loss_delta = loss_target - loss_prev
        loss_embedding = [self.scalar_embedder(loss_delta)]
        if self.absolute_loss_conditioning:
            loss_target_embedding = self.scalar_embedder(loss_target)
            loss_prev_embedding = self.scalar_embedder(loss_prev)
            loss_embedding.extend([loss_target_embedding, loss_prev_embedding])
        loss_embedding = torch.cat(loss_embedding, 1)
        return loss_embedding

    def forward(self, x, t, loss_target, loss_prev=None, x_prev=None):
        """
        Full G.pt forward pass.
        ----------------------------------------------
        N = batch size
        D = number of parameters
        ----------------------------------------------
        x: (N, D) tensor of noised updated parameters
        t: (N, 1) tensor indicating the diffusion timestep
        loss_target: (N, 1) tensor, the prompted (desired) loss/error/return
        loss_prev: (N, 1) tensor, loss/error/return obtained by x_prev
        x_prev: (N, D) tensor of starting parameters that are being updated
        ----------------------------------------------
        returns: (N, D) tensor of denoised updated parameters
        ----------------------------------------------
        """
        t_embedding = self.scalar_embedder(t)
        loss_embedding = self.embed_loss(loss_target, loss_prev)
        inp = [x, x_prev, t_embedding, loss_embedding]
        inp = torch.cat(inp, 1)
        output = self.decoder(inp)
        # Global residual connection:
        if self.use_global_residual:
            output = output + x_prev
        return output
