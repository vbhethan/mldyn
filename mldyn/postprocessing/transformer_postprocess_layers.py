import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PostProcessSelfAttentionLayer(nn.Module):
    """
    Implementation of the self attention layer that also returns the attention weights
    see mldyn.layers.layers.SelfAttentionLayer
    """

    def __init__(self, input_dimension, embed_size):
        super(PostProcessSelfAttentionLayer, self).__init__()
        self.input_dimension = input_dimension
        self.embed_size = embed_size

        # Query, Key, Value linear layers
        self.query = nn.Linear(input_dimension, embed_size)
        self.key = nn.Linear(input_dimension, embed_size)
        self.value = nn.Linear(input_dimension, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, input_dimension)

    def forward(self, x):

        # Compute the query, key, value tensors
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute the attention scores (Shape will be (N_particles, N_particles))
        attention_scores = torch.matmul(q, k.transpose(-2, -1) / self.embed_size**0.5)
        # Apply softmax to compute the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the output tensor
        out = torch.matmul(attention_weights, v)
        out = self.fc_out(out)

        return out, attention_weights


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


class PostProcessEncoderLayer(nn.Module):
    """
    Modification of the encoder layer to return the attention weights
    see mldyn.layers.layers.EncoderLayer
    Composition of the self-attention layer and feed forward neural network for an encoder layer
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PostProcessEncoderLayer, self).__init__()
        self.self_attention = PostProcessSelfAttentionLayer(d_model, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self Attention
        attention_output, attention_weights = self.self_attention(x)
        x = self.norm1(x + self.dropout(attention_output))

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights


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


class PostProcessCrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(PostProcessCrossAttentionLayer, self).__init__()
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
        attention_scores = torch.matmul(Q, K.transpose(-2, -1) / (self.d_model**0.5))

        # Softmax for weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self.fc_out(output)

        return output, attention_weights


class PostProcessDecoderLayer(nn.Module):

    def __init__(self, d_model, d_feedforward, dropout=0.0):
        super(PostProcessDecoderLayer, self).__init__()
        self.self_attention = PostProcessSelfAttentionLayer(d_model, d_model)
        self.cross_attention = PostProcessCrossAttentionLayer(d_model)
        self.feed_forward = FeedForward(d_model, d_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, memory):

        # Self Attention
        target2 = self.norm1(target)
        self_attention_output, self_attention_weights = self.self_attention(target2)
        target = target + self.dropout(self_attention_output)

        # Cross attention of target with the memory from the encoder output
        target2 = self.norm2(target)
        cross_attention_output, cross_attention_weights = self.cross_attention(
            target2, memory, memory
        )
        target = target + self.dropout(cross_attention_output)

        # Feed Forward NN
        target2 = self.norm3(target)
        target = target + self.dropout(self.feed_forward(target2))

        return target, self_attention_weights
