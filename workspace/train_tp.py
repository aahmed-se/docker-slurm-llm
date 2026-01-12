import os
import json
import time
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Config ---
DATA_DIR = '/data/data'
TEXT_FILE = os.path.join(DATA_DIR, 'tinyshakespeare.txt')
META_FILE = os.path.join(DATA_DIR, 'meta.json')
CKPT_DIR = '/data/checkpoints'

BATCH_SIZE = 4
EMBED_DIM = 64  
HIDDEN_DIM = 64 

def log(msg):
    """Helper to print with Node and Rank info"""
    rank = os.environ.get('RANK', '?')
    # Use multiple methods to detect actual node
    slurm_nodeid = os.environ.get('SLURM_NODEID', '?')
    slurm_node = os.environ.get('SLURMD_NODENAME', '?')
    hostname = os.environ.get('HOSTNAME', '?')
    actual_hostname = socket.gethostname()
    
    print(f"[Rank: {rank} | SLURM_NODEID: {slurm_nodeid} | SLURMD_NODENAME: {slurm_node} | Hostname: {actual_hostname}] {msg}")

def setup():
    """ Initialize Slurm Distributed Environment """
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    log(f"Initializing Process Group... (Master: {master_addr})")
    
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    log("Process Group Initialized. Ready to train.")
    return rank, world_size

class ShakespeareDataset(Dataset):
    def __init__(self, text_path, meta_path, seq_len=128):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.stoi = meta['stoi']
        
        with open(text_path, 'r', encoding='utf-8') as f:
            data = f.read()
            
        self.tokens = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        self.seq_len = seq_len
        torch.manual_seed(42)
        self.embedding_table = torch.randn(meta['vocab_size'], EMBED_DIM)

    def __len__(self):
        return 40 # Short dataset for demo

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx+self.seq_len]
        embedded_chunk = torch.nn.functional.embedding(chunk, self.embedding_table)
        return embedded_chunk.mean(dim=0)

class TPLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        self.partition_dim = output_dim // world_size
        self.weight = nn.Parameter(torch.randn(input_dim, self.partition_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight)

def train():
    rank, world_size = setup()
    
    if not os.path.exists(TEXT_FILE):
        log("Error: Data file not found.")
        return

    log("Loading Dataset...")
    dataset = ShakespeareDataset(TEXT_FILE, META_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TPLinear(EMBED_DIM, HIDDEN_DIM, world_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    log(f"Starting Training Loop for 10 Epochs. Dataset len: {len(dataset)}")

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Local computation
            local_out = model(batch)
            loss = local_out.mean()
            loss.backward()
            optimizer.step()
            
            # Log progress every few batches
            if batch_idx % 5 == 0:
                log(f"Epoch {epoch} | Processing Batch {batch_idx}/{len(dataloader)}")
            
            # Sync Loss for display
            reduced_loss = loss.clone().detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= world_size
            total_loss += reduced_loss.item()

        epoch_time = time.time() - start_time
        
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"--- [SUMMARY] Epoch {epoch} Finished in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} ---")
            
        # Checkpointing
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR, exist_ok=True)
            
        ckpt_path = f'{CKPT_DIR}/model_rank_{rank}_ep{epoch}.pt'
        torch.save(model.state_dict(), ckpt_path)
        log(f"Saved checkpoint: {ckpt_path}")

    log("TRAINING COMPLETE. Exiting.")
    dist.destroy_process_group()

if __name__ == '__main__':
    train()