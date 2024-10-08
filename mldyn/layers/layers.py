import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttentionLayer(nn.Module):
    """
    Implement the self-attention layer
    Input:
        - x: input tensor of shape (batch_size, sequence_length, input_dimension)

    Output:
        - output: output tensor of shape (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, input_dimension, attention_dimension):
        super(SelfAttentionLayer, self).__init__()
        self.input_dimension = input_dimension
        self.attention_dimension = attention_dimension

        # Query, Key, Value linear layers
        self.query = nn.Linear(input_dimension, attention_dimension)
        self.key = nn.Linear(input_dimension, attention_dimension)
        self.value = nn.Linear(input_dimension, attention_dimension)

        # Output linear layer
        self.fc_out = nn.Linear(attention_dimension, input_dimension)

    def forward(self, x):

        # Compute the query, key, value tensors
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute the attention scores (Shape will be #TODO: check)
        attention_scores = torch.matmul(q, k.transpose(-2, -1) / self.embed_size**0.5)
        # Apply softmax to compute the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the output tensor
        out = torch.matmul(attention_weights, v)
        out = self.fc_out(out)

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
    """
    Composition of the self-attention layer and feed forward neural network for an encoder layer
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttentionLayer(d_model, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self Attention
        attention_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attention_output))

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class ResidueEmbeddingLayer(nn.Module):

    def __init__(self, n_particles, input_state_dimension, d_model, n_particle_types):
        super(ResidueEmbeddingLayer, self).__init__()
        self.n_particles = n_particles
        self.d_model = d_model
        self.n_particle_types = n_particle_types

        # Particle State Embedding
        self.state_embedding = nn.Embedding(input_state_dimension, d_model)

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


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.d_model = d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Linear Transforms
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Compute attention scores
        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1) / torch.sqrt(self.d_model, dtype=torch.float32)
        )

        # Softmax for weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self.fc_out(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_feedforward, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttentionLayer(d_model, d_model)
        self.cross_attention = CrossAttentionLayer(d_model)
        self.feed_forward = FeedForward(d_model, d_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, memory):

        # Self Attention
        target2 = self.norm1(target)
        target = target + self.dropout(self.self_attention(target2))

        # Cross attention of target with the memory from the encoder output
        target2 = self.norm2(target)
        target = target + self.dropout(self.cross_attention(target2, memory, memory))

        # Feed Forward NN
        target2 = self.norm3(target)
        target = target + self.dropout(self.feed_forward(target2))

        return target
