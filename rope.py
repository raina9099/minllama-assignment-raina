from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, head_dim = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # print("Shape of query: ", query.shape) #(batch_size, seqlen, n_local_heads, self.head_dim)
    # print(query[0][0])
    # print("Shape of key: ", key.shape) #(batch_size, seqlen, n_local_heads, self.head_dim)
    
    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    # print("Shape of query_real: ", query_real.shape) #(batch_size, seqlen, n_local_heads, self.head_dim/2)
    # print(query_real[0][0])
    # print("Shape of query_imag: ", query_imag.shape) #(batch_size, seqlen, n_local_heads, self.head_dim/2)
    # print(query_imag[0][0])
    
    # batch_size, seqlen, n_local_heads, half_head_dim = query_real.shape

    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # print("Shape of key_real: ", key_real.shape) #(batch_size, seqlen, n_local_heads, self.head_dim/2)
    # print("Shape of key_imag: ", key_imag.shape) #(batch_size, seqlen, n_local_heads, self.head_dim/2)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    thetas = theta ** (-2 * torch.arange(0, head_dim//2, dtype=torch.float32) / head_dim) 
    
    # print("Shape of thetas: ", thetas.shape) #(self.head_dim/2)
    # print(thetas)
    # print("Shape of thetas.unsqueeze(0): ", thetas.unsqueeze(0).shape) #(1, self.head_dim/2)
    m_s = torch.arange(0, seqlen, dtype=torch.float32).unsqueeze(1)
    # print("Shape of m_s", m_s.shape) #(seqlen, 1)
    # print(m_s)
    m_thetas = thetas.unsqueeze(0) * m_s
    # print("Shape of m_thetas: ", m_thetas.shape) #(seqlen, self.head_dim/2)
    # print(m_thetas)
    # print("Shape of m_thetas.unsqueeze(1): ", m_thetas.unsqueeze(1).shape) #(1, 1, self.head_dim/2)
    # m_thetas = torch.cat((m_thetas, m_thetas), dim=1)
    # print("Shape of m_thetas after torch.cat: ", m_thetas.shape) #(seqlen, self.head_dim)
    stacked_m_thetas = torch.stack([m_thetas, m_thetas], dim=-1)
    # Reshape to flatten the last two dimensions
    m_thetas = stacked_m_thetas.view(*stacked_m_thetas.shape[:-2], -1)
    # print(m_thetas)
    sin_m_thetas = m_thetas.sin()
    cos_m_thetas = m_thetas.cos()
    # print("Shape of sin_m_thetas: ", sin_m_thetas.shape) #(seqlen, self.head_dim)
    # print(sin_m_thetas)
    # print("Shape of cos_m_thetas: ", cos_m_thetas.shape) #(seqlen, self.head_dim)
    # print(cos_m_thetas)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    query_stacked = torch.stack([-query_imag, query_real], dim=-1)
    # Reshape to flatten the last two dimensions
    query_dash = query_stacked.view(*query_stacked.shape[:-2], -1)
    
    # print("Shape of query_dash: ", query_dash.shape) #(batch_size, seqlen, n_local_heads, self.head_dim)
    # print(query_dash[0][0])
    query_out = query.transpose(1,2) * cos_m_thetas + query_dash.transpose(1,2) * sin_m_thetas
    # query_out = query * cos_m_thetas + query_dash * sin_m_thetas
    # print("query * cos_m_thetas")
    # print((query * cos_m_thetas)[0][0])
    # print("query_dash * sin_m_thetas")
    # print((query_dash * sin_m_thetas)[0][0])
    # print("Shape of query_out: ", query_out.shape) #(batch_size, n_local_heads, seqlen, self.head_dim)

    key_stacked = torch.stack([-key_imag, key_real], dim=-1)
    # Reshape to flatten the last two dimensions
    key_dash = key_stacked.view(*key_stacked.shape[:-2], -1)

    # print("Shape of key_dash: ", key_dash.shape) #(batch_size, seqlen, n_local_heads, self.head_dim)
    key_out = key.transpose(1,2) * cos_m_thetas + key_dash.transpose(1,2) * sin_m_thetas
    # print("Shape of key_out: ", key_out.shape)  #(batch_size, seqlen, n_local_heads, self.head_dim)

    # Return the rotary position embeddings for the query and key tensors
    return query_out.transpose(1,2), key_out.transpose(1,2)
