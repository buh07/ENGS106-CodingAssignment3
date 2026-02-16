import torch
try:
    from tqdm import tqdm  # fallback to standard tqdm if notebook widgets unavailable
except ImportError:
    from tqdm.notebook import tqdm
from torch.nn import functional as F
from dataset import getBatch
import pickle
import gc

def loss_func(logits,targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

# calculate the loss ｜
@torch.no_grad()
def validation(model,train_config):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            x_batch, targets = getBatch(split, train_config.train_dataloader, train_config.test_dataloader)
            with train_config.ctx:
                logits = model(x_batch)
                loss = loss_func(logits,targets)
            losses[k] = loss.item()
            del loss,logits
        out[split] = losses.mean()
        del losses
    model.train()
    return out

def train(model, train_config):
    pbar = tqdm(range(train_config.max_iters), miniters=100, mininterval=1, leave=True)
    pbar.set_postfix_str(f'Training Loss: {-1} Validation Loss: {-1}')
    for step in pbar:
        if step % train_config.eval_interval == 0 or step == train_config.max_iters - 1:
            losses = validation(model,train_config)
            train_loss = round(losses['train'].item(), 3)
            valid_loss = round(losses['valid'].item(), 3)
            pbar.set_postfix_str(f"Training Loss: {train_loss} Validation Loss: {valid_loss}")
            pbar.refresh()
            del losses
            gc.collect()
            torch.cuda.empty_cache()
        xb, targets = getBatch('train', train_config.train_dataloader, train_config.test_dataloader)
        with train_config.ctx:
            logits = model(xb)
            loss = loss_func(logits,targets)
        train_config.scaler.scale(loss).backward()
        if train_config.grad_clip != 0.0:
            train_config.scaler.unscale_(train_config.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        train_config.scaler.step(train_config.optimizer)
        train_config.scaler.update()
        train_config.optimizer.zero_grad(set_to_none=True)
        del logits, loss


def generate(model, idx, max_new_tokens=100, temperature=1.0, top_k=None):
    # idx is (B,T) array of indices in the current context |
    for _ in range(max_new_tokens):
        # Crop idx to the max size of our positional embeddings table |
        idx_crop = idx[:, -model.context_length:]
        # Get predictions |
        logits = model(idx_crop)
        # Get the last time step from logits where the dimensions of the logits are (B,T,C) |
        logits = logits[:, -1, :] / temperature # Divide by temperature |
        # optionally crop the logits to only the top k options |
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to get probabilities |
        probs = F.softmax(input=logits, dim=-1)
        # Sample from the probabilities' distribution. |
        idx_next = torch.multinomial(input=probs, num_samples=1)
        # Append the sampled indexes idx_next to idx |
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def inference(model, inference_config):
    with open(inference_config.meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    start_ids = encode(inference_config.start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=inference_config.device)[None, ...])

    # run generation ｜
    with torch.no_grad():
        y = generate(model, x, max_new_tokens=inference_config.max_new_tokens, temperature=inference_config.temperature, top_k=inference_config.top_k)
        return decode(y[0].tolist())
