"""
Embeddings for characters and tokens, as well as concatenation of embeddings.
"""

import torch
from torch import nn


class CharEmbedding(nn.Module):
    """
    Convolutional character embedding.
    """

    PAD_CHAR = 0

    def __init__(self, num_embeddings, embedding_dim,
                 num_filters,
                 filter_sizes=(2, 3, 4, 5),
                 activation_class=nn.ReLU,
                 output_dim=None):
        super(CharEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.activation = activation_class()

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # convolution step
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim,
                       out_channels=num_filters,
                       kernel_size=_size)
             for _size in filter_sizes])

        base_output_dim = num_filters*len(self.layers)
        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(base_output_dim, output_dim)
        else:
            self.output_dim = base_output_dim
            self.projection = None

        return

    def forward(self, characters):
        """
        Encode the tokens based on the characters.
        """
        batch, num_tokens, num_chars = characters.size()
        char_emb = self.embeddings(
            characters.view(batch*num_tokens, num_chars))
        # char_emb: [b*t, c, emb]

        # Mask for padding.
        mask = (characters != self.PAD_CHAR)
        mask = mask.view(batch*num_tokens, num_chars, 1).float()
        char_emb = char_emb * mask

        # Transpose to match Conv1d, which expects [b*t, emb, c]
        char_emb = torch.transpose(char_emb, 1, 2)

        # Apply the convolution, and perform max pooling.
        # _outputs: [[b*t, num_filter]]
        _outputs = [
            self.activation(_layer(char_emb)).max(dim=2)[0]
            for _layer in self.layers]

        if len(_outputs) > 1:
            output = torch.cat(_outputs, dim=1)
        else:
            output = _outputs[0]
        # output: [batch*t, num_filter*len(filter_size)]

        if self.projection:
            output = self.projection(output)
        # output: [batch*t, output_dim]

        # Finally, unravel the time dimension.
        output = output.view(batch, num_tokens, self.output_dim)

        return output


class TokenEmbedding(nn.Module):
    """
    Token embedding.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 output_dim=None, static=True):
        super(TokenEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        if static:
            for param in self.embeddings.parameters():
                param.requires_grad = False

        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(embedding_dim, output_dim)
        else:
            self.output_dim = embedding_dim
            self.projection = None

        return

    def forward(self, tokens):
        """
        Get the embeddings for tokens
        """
        batch, num_tokens = tokens.size()
        token_emb = self.embeddings(tokens)
        # token_emb: [b, t, emb]
        if self.projection:
            token_emb = self.projection(token_emb.view(
                batch*num_tokens, self.embedding_dim))
            token_emb = token_emb.view(batch, num_tokens, self.output_dim)
        # output: [batch, t, output_dim]

        return token_emb


class StaticTokenEmbedding(TokenEmbedding):
    """
    Convenience class to create static token embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(StaticTokenEmbedding, self).__init__(
            num_embeddings, embedding_dim, None, True)
        return


class UpdatedTokenEmbedding(TokenEmbedding):
    """
    Convenience class to create updated token embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(UpdatedTokenEmbedding, self).__init__(
            num_embeddings, embedding_dim, None, False)
        return


class CatEmbedding(nn.Module):
    """
    Concatenate embeddings together, possibly with a final projection.
    """

    def __init__(self, embeddings, output_dim=None):
        super(CatEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)

        embedding_dim = sum(_emb.output_dim for _emb in self.embeddings)
        if output_dim:
            self.output_dim = output_dim
            self.projection = nn.Linear(embedding_dim, output_dim)
        else:
            self.output_dim = embedding_dim
            self.projection = None
        return

    def forward(self, features):
        if len(features) != len(self.embeddings):
            raise ValueError('CatEmbedding: mismatch between number of'
                             ' features and number of embedding',
                             len(features), len(self.embeddings))

        combined = torch.cat([
            _e(_f)
            for _f, _e in
            zip(features, self.embeddings)],
            dim=2)

        if self.projection:
            combined = self.projection(combined)

        return combined