import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLayer(nn.Module):
    """
    Unified attention layer that can handle both self-attention and cross-attention

    For self-attention: pass the same tensor as query, key, and value
    For cross-attention: pass different tensors for query vs key/value

    Input:
        - query: tensor of shape (batch_size, query_length, input_dimension)
        - key: tensor of shape (batch_size, key_length, input_dimension)
        - value: tensor of shape (batch_size, key_length, input_dimension)

    Output:
        - output: tensor of shape (batch_size, query_length, input_dimension)
    """

    def __init__(
        self, input_dimension, embed_size, num_heads=8, return_attention=False
    ):
        super(AttentionLayer, self).__init__()
        self.input_dimension = input_dimension
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.return_attention = return_attention
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.head_dim = embed_size // num_heads

        # Query, Key, Value linear layers
        self.query = nn.Linear(input_dimension, embed_size)
        self.key = nn.Linear(input_dimension, embed_size)
        self.value = nn.Linear(input_dimension, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, input_dimension)

    def forward(self, query, key=None, value=None):
        # If key is None, use query (self-attention case)
        if key is None:
            key = query
        # If value is None, use key
        if value is None:
            value = key

        batch_size = query.shape[0]
        query_length = query.shape[1]
        key_length = key.shape[1]

        # Compute query, key, value and split into heads
        q = self.query(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        k = self.key(key).view(batch_size, key_length, self.num_heads, self.head_dim)
        v = self.value(value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        )

        # Transpose to get dimensions: (batch_size, num_heads, seq_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores for all heads simultaneously
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, v)

        # Reshape back: (batch_size, query_length, embed_size)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_length, self.embed_size)
        )

        # Final linear projection
        out = self.fc_out(out)

        if self.return_attention:
            # Average attention weights across heads
            attention_weights = attention_weights.mean(dim=1)
            return out, attention_weights
        return out


class FeedForward(nn.Module):
    """
    Feed forward neural network for the transformer encoder
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer that uses self-attention"""

    def __init__(
        self, d_model, d_feedforward, dropout=0.0, num_heads=8, return_attention=False
    ):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(
            d_model, d_model, num_heads=num_heads, return_attention=return_attention
        )
        self.feed_forward = FeedForward(d_model, d_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.return_attention = return_attention

    def forward(self, x):
        # Self Attention
        x2 = self.norm1(x)
        if self.return_attention:
            attention_output, attention_weights = self.attention(x2)
            x = x + self.dropout(attention_output)
        else:
            x = x + self.dropout(self.attention(x2))

        # Feed Forward
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))

        if self.return_attention:
            return x, attention_weights
        return x


class ResidueEmbeddingLayer(nn.Module):

    def __init__(self, n_particles, input_state_dimension, d_model, n_particle_types):
        super(ResidueEmbeddingLayer, self).__init__()
        self.n_particles = n_particles
        self.d_model = d_model
        self.n_particle_types = n_particle_types

        # Particle State Embedding
        self.state_embedding = nn.Linear(input_state_dimension, d_model)

        # Positional Embedding
        self.pos_encoding = self.create_positional_encoding()

        # Particle Type Embedding
        self.type_embedding = nn.Embedding(n_particle_types, d_model)

    def create_positional_encoding(self):
        pos_encoding = torch.zeros(self.n_particles, self.d_model)
        position = torch.arange(0, self.n_particles, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )  # Formula / values from the Attention is all you need paper...
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, particle_states, particle_types):
        # Particle states shape: (batch_size, n_particles, input_state_dimension)
        # Particle types shape: (batch_size, n_particles)

        batch_size = particle_states.shape[0]

        # Embed the particle states
        state_embedding = self.state_embedding(particle_states)

        # Add the positional encoding
        positional_encoding = self.pos_encoding.repeat(batch_size, 1, 1).to(
            particle_states.device
        )
        state_position_embedding = state_embedding + positional_encoding

        # Embed the particle types
        type_embedding = self.type_embedding(particle_types)

        # Combine the embeddings
        residue_embedding = state_position_embedding + type_embedding

        return residue_embedding


class DecoderLayer(nn.Module):
    """Decoder layer with self-attention and cross-attention"""

    def __init__(
        self, d_model, d_feedforward, dropout=0.0, num_heads=8, return_attention=False
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(
            d_model, d_model, num_heads=num_heads, return_attention=return_attention
        )
        self.cross_attention = AttentionLayer(
            d_model, d_model, num_heads=num_heads, return_attention=return_attention
        )
        self.feed_forward = FeedForward(d_model, d_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.return_attention = return_attention

    def forward(self, target, memory):
        # Self Attention
        target2 = self.norm1(target)
        if self.return_attention:
            self_attention_output, self_attention_weights = self.self_attention(target2)
            target = target + self.dropout(self_attention_output)
        else:
            target = target + self.dropout(self.self_attention(target2))

        # Cross attention
        target2 = self.norm2(target)
        if self.return_attention:
            cross_attention_output, cross_attention_weights = self.cross_attention(
                query=target2, key=memory, value=memory
            )
            target = target + self.dropout(cross_attention_output)
        else:
            target = target + self.dropout(
                self.cross_attention(query=target2, key=memory, value=memory)
            )

        # Feed Forward
        target2 = self.norm3(target)
        target = target + self.dropout(self.feed_forward(target2))

        if self.return_attention:
            return target, self_attention_weights, cross_attention_weights
        return target
