# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import EPS


def batch_embedding_lookup(embeddings, indices):
    """
    Look up from a batch of embedding matrices.
    :param embeddings: (batch_size, num_words, embedding_size)
    :param indices: (batch_size, num_inds)
    :return:
    """
    shape = embeddings.shape
    batch_size = shape[0]
    num_words = shape[1]
    embed_size = shape[2]

    offset = torch.reshape(torch.arange(batch_size) * num_words, (batch_size, 1)).to(dtype=indices.dtype,
                                                                                     device=indices.device)
    flat_embeddings = torch.reshape(embeddings, (-1, embed_size))
    flat_indices = torch.reshape(indices + offset, (-1,))
    embeds = torch.reshape(F.embedding(flat_indices, flat_embeddings), (batch_size, -1, embed_size))
    return embeds


def mask_softmax(input_tensor, mask, dim):
    """
    Softmax with mask to input tensor.
    :param input_tensor: Tensor with any shape
    :param mask: same shape with input tensor
    :param dim: a dimension along which softmax will be computed
    :return:
    """
    exps = torch.exp(input_tensor)
    masked_exps = exps * mask.to(exps.dtype)
    masked_sums = masked_exps.sum(dim, keepdim=True) + EPS
    return masked_exps / masked_sums


class Attn(nn.Module):
    def __init__(self, method, encode_hidden_size, decode_hidden_size):
        super(Attn, self).__init__()
        if method.lower() not in ["dotted", "general", "concat"]:
            raise RuntimeError("Attention methods should be dotted, general or concat but get {}!".format(method))
        if method.lower() == "dotted" and encode_hidden_size != decode_hidden_size:
            raise RuntimeError("In dotted attention, the encode_hidden_size should equal to decode_hidden_size.")

        self.method = method.lower()
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size

        if self.method == "general":
            self.attn = nn.Linear(self.encode_hidden_size, self.decode_hidden_size)
        elif self.method == "concat":
            self.attn = nn.Sequential(
                nn.Linear((self.encode_hidden_size + self.decode_hidden_size),
                          (self.encode_hidden_size + self.decode_hidden_size) // 2),
                nn.Tanh(),
                nn.Linear((self.encode_hidden_size + self.decode_hidden_size) // 2, 1)
            )

    def forward(self, encode_outputs, decode_state):
        """
        :param encode_outputs: (batch, output_length, encode_hidden_size)
        :param decode_state: (batch, decode_hidden_size)
        :return: energy: (batch, output_length)
        """
        output_length = encode_outputs.size(1)
        if self.method == "concat":
            decode_state_temp = decode_state.unsqueeze(1)
            decode_state_temp = decode_state_temp.expand(-1, output_length, -1)
            cat_encode_decode = torch.cat([encode_outputs, decode_state_temp], 2)
            energy = self.attn(cat_encode_decode).squeeze(-1)
        elif self.method == "general":
            decode_state_temp = decode_state.unsqueeze(1)
            mapped_encode_outputs = self.attn(encode_outputs)
            energy = torch.sum(decode_state_temp * mapped_encode_outputs, 2)
        else:
            decode_state_temp = decode_state.unsqueeze(1)
            energy = torch.sum(decode_state_temp * encode_outputs, 2)
        return energy
