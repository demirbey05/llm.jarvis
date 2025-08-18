from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.LazyLinear(d_out,bias=qkv_bias)
        self.W_key = nn.LazyLinear(d_out,bias=qkv_bias)
        self.W_value = nn.LazyLinear(d_out,bias=qkv_bias)

    def forward(self, inputs):
        query_matrix = self.W_query(inputs)
        key_matrix = self.W_key(inputs)
        value_matrix = self.W_value(inputs)
        
        attn_scores = query_matrix @ key_matrix.T / ((key_matrix.shape[-1]) ** 0.5)
        attn_matrix = nn.functional.softmax(attn_scores, dim=-1)
        context_matrix = attn_matrix @ value_matrix
        return context_matrix


class MultiHeadAttention(nn.Module):
  def __init__(self,d_out,num_heads,context_length,qkv_bias,dropout=0.0):
    super().__init__()

    assert (d_out % num_heads == 0)

    self.W_queries = nn.LazyLinear(d_out,bias=qkv_bias)
    self.W_keys = nn.LazyLinear(d_out,bias=qkv_bias)
    self.W_values = nn.LazyLinear(d_out,bias=qkv_bias)
    self.out_proj = nn.LazyLinear(d_out)
    self.dropout = nn.Dropout(dropout)

    self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
    self.head_dim = d_out // num_heads
    self.num_heads = num_heads
    self.d_out = d_out


  def forward(self,inputs):
    # inputs has size (batch_size,context_length,d_in)
    # Get Queries, Keys and Values

    batch_size, context_length, d_in = inputs.shape

    queries = self.W_queries(inputs) # Size of (batch,context_length,d_out)
    keys = self.W_keys(inputs)  # Size of (batch,context_length,d_out)
    values = self.W_values(inputs)  # Size of (batch,context_length,d_out)

    # We should divide queries,keys and values
    queries = queries.view(queries.shape[0],queries.shape[1],self.num_heads,-1)
    keys = keys.view(keys.shape[0],keys.shape[1],self.num_heads,-1)
    values = values.view(values.shape[0],values.shape[1],self.num_heads,-1)

    # Transpose to get 
    queries = queries.transpose(1,2)# (batch_size,num_heads,context_len,d_out)
    keys = keys.transpose(1,2)# (batch_size,num_heads,context_len,d_out)
    values = values.transpose(1,2) # (batch_size,num_heads,context_len,d_out)

    attn_scores = queries @ keys.transpose(-1,-2) / ((keys.shape[-1]) ** 0.5) # (batch_size,num_heads,context_len,context_len) = batch_size * num_heads amount matrix
    mask_adapted = self.mask[:context_length,:context_length]
    attn_scores.masked_fill_(mask_adapted.bool(),-torch.inf) 

    attn_matrix = nn.functional.softmax(attn_scores,dim=-1)
    attn_matrix = self.dropout(attn_matrix)

    context_matrix = attn_matrix @ values # # (batch_size,num_heads,context_len,head_dim)
    context_matrix = context_matrix.transpose(1, 2) # (batch_size,context_len,num_heads,head_dim)
    context_vec = context_matrix.contiguous().view(
      batch_size, context_length, self.d_out
    )

    context_vec = self.out_proj(context_vec)

    return context_vec