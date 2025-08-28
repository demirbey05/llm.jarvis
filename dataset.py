from torch.utils.data import Dataset,DataLoader
import torch
import tiktoken
class GPTDataset(Dataset):
  def __init__(self,text,tokenizer,max_len=4,stride=1):
    self.input_ids = []
    self.target_ids = []

    tokenized_data = tokenizer.encode(text)

    for i in range(0,len(tokenized_data) - max_len,stride):
      self.input_ids.append(torch.tensor(tokenized_data[i:i+max_len]))
      self.target_ids.append(torch.tensor(tokenized_data[i+1:i+max_len+1]))


  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self,idx):
    return self.input_ids[idx],self.target_ids[idx]


def create_dataloader(text,batch_size=4,shuffle=True,max_len=256,stride=128,drop_last=True):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDataset(text,tokenizer,max_len=max_len,stride=stride)
  dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
  return dataloader



def text_to_token_ids(text:str, tokenizer:tiktoken.Encoding):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0)
  return tokenizer.decode(flat.tolist())
