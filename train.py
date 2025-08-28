from dataset import text_to_token_ids,token_ids_to_text
import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch) #(batch_size,context_len,vocab_size)

    # we will use torch.nn.functional 
    # we must do some dimensional arrangements
    # target has size of (batch_size,context_len)
    # logits has size of #(batch_size,context_len,vocab_size)
    # We must transform targets to (batch_size * context_len)
    # We must transform logits to (batch_size * context_len, vocab_size)
    # Look Example small for cell below

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        # no data
        return float("nan")
    elif num_batches == None:
        # if num_batches is 0 then whole data
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader),num_batches)
    

    for i,(inp,target) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(inp, target, model, device)
            total_loss = loss.item()
        else:
            break
    return total_loss

def generate_text_simple(model, idx,max_new_tokens, context_size):
  #idx has size of (batch, n_tokens)
  for _ in range(max_new_tokens):
    # if we exceed context_size, we will use the last context_size tokens
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)
      # logits has size of (batch, context_size, vocab_size)
      logits = logits[:, -1, :]
      # for next token prediction get the last prediction
      # logits has size of (batch,1, vocab_size)
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
  print("Generated Text's token length:",idx.numel())
  return idx

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_text = generate_text_simple(model, encoded, max_new_tokens=100, context_size=context_size)
    
    decoded_text = token_ids_to_text(generated_text, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, 
train_loader,
val_loader,
optimizer,
device,
num_epochs,
eval_freq,
eval_iter,
start_context,
tokenizer):

    train_losses,val_losses, track_tokens_seen = [],[],[]
    tokens_seen, global_step = 0, -1

  
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inputs.numel()
            global_step += 1
        
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch}, Step {global_step}, Train Loss {train_loss}, Val Loss {val_loss}")
            
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen   
      
      