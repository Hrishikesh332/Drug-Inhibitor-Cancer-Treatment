import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from src.models.architectures import SynergyModel
from src.models.metrics import calc_pearson, calc_spearman
from src.utils.data_loader import load_data
from src.utils.visualization import plot_metrics, plot_preds


def train_epoch(model, loader, opt, crit, dev):
    model.train()
    loss_sum = 0.0
    pear_sum = 0.0
    
    all_pred = []
    all_true = []
    
    for x, y in loader:
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        pear = calc_pearson(out, y)
        
        loss.backward()
        opt.step()
        
        loss_sum += loss.item() * x.size(0)
        pear_sum += pear.item() * x.size(0)
        
        all_pred.append(out.detach())
        all_true.append(y.detach())
    
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    spear = calc_spearman(all_pred, all_true)
    
    avg_loss = loss_sum / len(loader.dataset)
    avg_pear = pear_sum / len(loader.dataset)
    
    return avg_loss, avg_pear, spear.item()

def eval_model(model, loader, crit, dev):
    model.eval()
    loss_sum = 0.0
    pear_sum = 0.0
    
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss = crit(out, y)
            pear = calc_pearson(out, y)
            
            loss_sum += loss.item() * x.size(0)
            pear_sum += pear.item() * x.size(0)
            
            all_pred.append(out)
            all_true.append(y)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    spear = calc_spearman(all_pred, all_true)
    
    avg_loss = loss_sum / len(loader.dataset)
    avg_pear = pear_sum / len(loader.dataset)
    
    return avg_loss, avg_pear, spear.item()

def train_model(cfg, tr_dl, ts_dl,sc, in_dim):
    fold = int(cfg.get('fold', 1))
    arch = str(cfg.get('arch', 'std'))
    batch = int(cfg.get('batch', 100))
    lr = float(cfg.get('lr', 1e-4)) 
    epochs = int(cfg.get('epochs', 2000))
    drop = float(cfg.get('drop', 0.3))
    log_every = int(cfg.get('log_every', 1))
    use_wb = bool(cfg.get('use_wb', True))
    wb_proj = str(cfg.get('wb_proj', 'synergy'))
    wb_run = str(cfg.get('wb_run', f'fold{fold}_{arch}'))
    data_dir = str(cfg.get('data_dir', 'data'))
    out_dir = str(cfg.get('out_dir', 'outputs'))
    
    #print("Configuration parameters:")
    #print(f"  fold: {fold} (type: {type(fold)})")
    #print(f"  arch: {arch} (type: {type(arch)})")
    #print(f"  batch: {batch} (type: {type(batch)})")
    #print(f"  lr: {lr} (type: {type(lr)})")
    #print(f"  epochs: {epochs} (type: {type(epochs)})")
    #print(f"  drop: {drop} (type: {type(drop)})")
    
    model_dir = os.path.join(out_dir, 'models')
    log_dir = os.path.join(out_dir, 'logs')
    res_dir = os.path.join(out_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    start = time.time()
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using: {dev}")
    
    if use_wb:
        wandb.init(project=wb_proj, name=wb_run, 
                  config={
                      'lr': lr,
                      'epochs': epochs,
                      'batch': batch,
                      'arch': arch,
                      'fold': fold,
                      'drop': drop
                  })
    
    if tr_dl is None or ts_dl is None or sc is None or in_dim is None:
        x_tr, y_tr, x_ts, y_ts, sc, tr_dl, ts_dl = load_data(cfg, data_dir, fold, batch=batch)
        #print(f"Train shape: {x_tr.shape}, Test shape: {x_ts.shape}")

    model = SynergyModel(in_dim=in_dim, arch=arch, drop=drop).to(dev)
    crit = nn.MSELoss()
    
    #print(f"Creating optimizer with learning rate: {lr} (type: {type(lr)})")
    opt = optim.Adam(model.parameters(), lr=lr)
    
    try:
        test_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt, 
            mode='min', 
            factor=0.5,
            patience=50,
            verbose=True
        )
        sched = test_scheduler
        #print("Using scheduler with verbose=True")
    except TypeError:
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='min',
            factor=0.5,
            patience=50
        )
        #print("Using scheduler without verbose parameter")
    
    #print(model)
    
    tr_loss = []
    tr_pear = []
    tr_spear = []
    val_loss = []
    val_pear = []
    val_spear = []
    best_val = -1.0
    step = 0
    
    #print("Starting training...")
    for e in tqdm(range(1, epochs + 1)):
        tl, tp, ts = train_epoch(model, tr_dl, opt, crit, dev)
        vl, vp, vs = eval_model(model, ts_dl, crit, dev)
        
        tr_loss.append(tl)
        tr_pear.append(tp)
        tr_spear.append(ts)
        val_loss.append(vl)
        val_pear.append(vp)
        val_spear.append(vs)
        
        sched.step(vl)
        step += 1
        
        if vs > best_val:
            best_val = vs
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_f{fold}.pt'))
        
        if use_wb:
            wandb.log({
                'epoch': e,
                'step': step,
                'train_loss': tl,
                'train_pearson': tp,
                'train_spearman': ts,
                'val_loss': vl,
                'val_pearson': vp,
                'val_spearman': vs,
                'lr': opt.param_groups[0]['lr']
            })
        
        if e % log_every == 0:
            print(f"Epoch {e}/{epochs} - "
                  f"TL: {tl:.4f}, TP: {tp:.4f}, TS: {ts:.4f}, "
                  f"VL: {vl:.4f}, VP: {vp:.4f}, VS: {vs:.4f}")
    
    torch.save(model.state_dict(), os.path.join(model_dir, f'final_f{fold}.pt'))
    
    plot_metrics(
        tr_loss, val_loss, 
        tr_pear, val_pear, 
        tr_spear, val_spear,
        os.path.join(res_dir, f'metrics_f{fold}.png')
    )
    
    if use_wb:
        wandb.log({"metrics": wandb.Image(os.path.join(res_dir, f'metrics_f{fold}.png'))})
    
    np.savetxt(os.path.join(log_dir, f'tr_pear_f{fold}.csv'), np.array(tr_pear), delimiter=',')
    np.savetxt(os.path.join(log_dir, f'val_pear_f{fold}.csv'), np.array(val_pear), delimiter=',')
    np.savetxt(os.path.join(log_dir, f'tr_spear_f{fold}.csv'), np.array(tr_spear), delimiter=',')
    np.savetxt(os.path.join(log_dir, f'val_spear_f{fold}.csv'), np.array(val_spear), delimiter=',')

    final_loss, final_pear, final_spear = eval_model(model, ts_dl, crit, dev)
    
    #print(f"Final Test - Loss: {final_loss:.4f}, Pearson: {final_pear:.4f}, Spearman: {final_spear:.4f}")
    
    if use_wb:
        wandb.log({
            'final_test_loss': final_loss,
            'final_test_pearson': final_pear,
            'final_test_spearman': final_spear
        })
        
        wandb.save(os.path.join(model_dir, f'final_f{fold}.pt'))
        wandb.save(os.path.join(model_dir, f'best_f{fold}.pt'))


        
        wandb.finish()
    
    with open(os.path.join(res_dir, f'summary_f{fold}.txt'), 'w') as f:
        f.write(f"Arch: {arch}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Time: {time.time() - start:.2f} seconds\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Final Pearson: {final_pear:.4f}\n")
        f.write(f"Final Spearman: {final_spear:.4f}\n")
        f.write(f"Best Val Spearman: {best_val:.4f}\n")
    
    #print(f"Training done in {time.time() - start:.2f} seconds")
    
    return {
        'model': model,
        'test_loss': final_loss,
        'test_pearson': final_pear,
        'test_spearman': final_spear,
        'best_val_spearman': best_val,
        'scaler': sc
    }

