import torch
import torch.nn as nn

# define the local self-attention layer
class LocalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        # define the basic parameters
        super(LocalSelfAttention, self).__init__()
        self.embed_size = embed_size  # the dimension of the input data
        self.heads = heads  # the number of the heads, which is the number of the attention
        self.window_size = window_size  # the size of the window
        # self.overlap_size = 0  # the size of the overlap
        self.head_dim = embed_size // heads  # the dimension of the head
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        # define the linear transformation layers
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)  # the output layer

    def forward(self, values, keys, queries, mask):
        # get the batch size and the length of the input data
        N, value_len, key_len, query_len = queries.shape[0], values.shape[1], keys.shape[1], queries.shape[1]
        # apply the linear transformation and reshape the data
        values = self.values(values.view(N, value_len, self.heads, self.head_dim))
        keys = self.keys(keys.view(N, key_len, self.heads, self.head_dim))
        queries = self.queries(queries.view(N, query_len, self.heads, self.head_dim))
        full_attention = torch.zeros(N, query_len, self.heads, self.head_dim, device=values.device)

        for i in range(0, query_len, self.window_size):
            window_end = min(i + self.window_size, query_len)
            # get the current window's queries, keys and values
            local_q = queries[:, i:window_end, :, :]
            local_k = keys[:, max(0, i - self.overlap_size):min(key_len, window_end + self.overlap_size), :, :]
            local_v = values[:, max(0, i - self.overlap_size):min(value_len, window_end + self.overlap_size), :, :]
            # compute the energies( the scores of the attention which is not be normalized)
            energies = torch.einsum("nqhd,nkhd->nhqk", [local_q, local_k])
            # the actual local_k size
            local_k_size = min(window_end + self.overlap_size, key_len) - max(0, i - self.overlap_size)
            if mask is not None:
                # adjust the mask size to adapt the current window size
                local_mask = mask[:, i:window_end].unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, window_size, 1]
                # adjust the shape to [batch_size, heads, window_size, actual_local_k_size]
                local_mask = (local_mask.expand(N, self.heads, -1, local_k_size))
                # use mask to ignore the padding values by setting the energies to a very small value
                energies = energies.masked_fill(local_mask == 0, float("-1e20"))
            # calculate the normalized attention scores and get the weighted output
            attention_local = torch.softmax(energies / (self.embed_size ** (1 / 2)), dim=3)
            out_local = torch.einsum("nhql,nlhd->nqhd", [attention_local, local_v])
            full_attention[:, i:window_end, :, :] \
                = out_local[:, :min(self.window_size, window_end - i), :, :]
        # reshape and pass through the last linear layer
        out = full_attention.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


# define the transformer block as the single layer of the transformer
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, window_size):
        super(TransformerBlock, self).__init__()
        # initialize the local self-attention layer
        self.attention = LocalSelfAttention(embed_size, heads, window_size)
        # layer normalization layers for output normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # feed forward network within the transformer block
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # compute the local self-attention
        attention = self.attention(value, key, query, mask)
        # apply dropout and layer normalization after adding the input (residual connection)
        # attention + query means the residual connection
        x = self.dropout(self.norm1(attention + query))
        # pass through the feed-forward network
        forward = self.feed_forward(x)
        # another dropout and normalization step
        out = self.dropout(self.norm2(forward + x))
        return out


# define the transformer model
class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, window_size
                 , input_size, num_classes):
        super(Transformer, self).__init__()
        # initial the basic parameters
        self.embed_size = embed_size
        self.device = device
        # create a list of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    window_size=window_size,
                )
                for _ in range(num_layers)
            ]
        )
        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # output linear layer to project to the number of classes
        self.fc_out = nn.Linear(embed_size, num_classes)
        # linear layer to project input features to the embedding size
        self.embedding = nn.Linear(input_size, embed_size)

    def forward(self, x, mask):
        # Embedding the input and applying dropout
        x = self.embedding(x)
        # this is a simple way to add noise to the input data, you can use other methods or close it
        if self.training:  # add noise during training
            noise = torch.randn_like(x) * 1e-2  # 1e-2 is the standard deviation
            x = x + noise  # add noise to the input data
        out = self.dropout(x)
        # Pass through each transformer block
        for layer in self.layers:
            out = layer(out, out, out, mask)
        # Output layer to get the final predictions
        out = self.fc_out(out)
        return out
