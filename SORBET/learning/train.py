from dataclasses import dataclass
from typing import List
import time
from tqdm import tqdm as tqdm

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score 
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from .dataset import TorchOmicsDataset 

@dataclass
class ModelParameters:
    epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int = -1
    validate_step: int = 2
    restarts: int = 5

# NOTE: Used in hyperparamter_optimization.py to automatically detect / suggest model paramters
_training_hyperparamters_types = [
        ('epochs', int, False),
        ('learning_rate', float, True),
        ('batch_size', int, False),
        ('validate_step', int, False),
        ('restarts', int, False)
        ]

def train_model(
        model_type, model_init_params, 
        dataset: List[TorchOmicsDataset],
        optimizer_type: torch.optim = torch.optim.Adam,
        epochs: int = 200, learning_rate: float = 1e-3, batch_size: int = -1, validate_step: int = 2, restarts: int = 5,
        progressbar: bool = False, print_lvl: int = 0, tensorboard_dir: str = None, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
    
    params = ModelParameters(epochs, learning_rate, batch_size, validate_step, restarts)

    if len(dataset) == 3:
        train, val, test = dataset
        cross_validate = True
    elif len(dataset) == 2:
        train, test = dataset
        cross_validate = False
    else:
        raise ValueError(f'Passed datasets have unhandled length. Assumes training structure from lengths 2 or 3. Does not handle passed length: {len(dataset)}') 
    
    batch_size = params.batch_size if params.batch_size > 0 else len(train)
    
    best_model, best_loss = None, np.inf
    best_train_losses, best_val_losses = None, None
    n_restarts = params.restarts if params.restarts >= 1 else 1
    for restart in range(n_restarts):
        tensorboard_descriptor = f'/restart_{restart}_{str(time.time()).replace(".","_")}'
        logger = SummaryWriter(log_dir=tensorboard_dir + tensorboard_descriptor) if tensorboard_dir is not None else None
        train_loss, val_loss = list(), list()
        
        model = model_type(**model_init_params) 
        model.to(device)
        optimizer = optimizer_type(model.parameters(), lr=params.learning_rate)
        for epoch in tqdm(range(params.epochs), disable=(not progressbar)):
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            epoch_losses = _train_step(model, train_loader, optimizer, device, epoch, logger=logger)
            
            if print_lvl >= 3 and (epoch + 1) % print_step == 0:
                print(f'Epoch {epoch+1} with train loss {np.mean(epoch_losses)} (std: {np.mean(epoch_losses)})')
            
            if cross_validate and (epoch + 1) % params.validate_step == 0:
                val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
                preds, labs, loss = _validate_step(model, val_loader, device, epoch, logger=logger)
                val_loss.append((epoch+1, loss))
                if print_lvl >= 2:
                    print(f'CV (Epoch {epoch+1}) with cross-validation loss {loss} (AUROC: {roc_auc_score(labs, preds):.3f}, Avg. Prec.: {average_precision_score(labs, preds):.3f})')

            train_loss.append(epoch_losses)

        upd_cv = cross_validate and val_loss[-1][1] < best_loss
        upd_tr = not cross_validate and np.mean(train_loss[-1]) < best_loss
        #if np.mean(train_loss[-1]) < best_loss:
        if upd_cv or upd_tr:
            best_model, best_loss = model, np.mean(train_loss[-1])
            best_train_losses, best_val_losses = train_loss, val_loss

    model = best_model
    train_loss, val_loss = best_train_losses, best_val_losses

    if print_lvl >= 1 and cross_validate:
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        preds, labs, loss = _validate_step(model, val_loader, torch.sigmoid, device, epoch, logger=None)
        val_loss.append((epoch+1, loss))
        print(f'Final CV with cross-validation loss {loss} (AUROC: {roc_auc_score(labs, preds):.3f}, Avg. Prec.: {average_precision_score(labs, preds):.3f})')

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    preds, labs = _test_step(model, test_loader, device)
    test_auroc = roc_auc_score(labs, preds) 
    test_avg_prec = average_precision_score(labs, preds)
    if print_lvl >= 1:
        print(f'Model completed training. (AUROC: {test_auroc:.3f}, Avg. Prec.: {test_avg_prec:.3f})')


    if logger is not None:
        hparams_dct = {
                "learning_rate": params.learning_rate,
                "batch_size": batch_size,
                "epochs": params.epochs
                }

        results_dct = {
                "FINAL_train_loss": np.mean(train_loss[-1]),
                "FINAL_test_auroc": test_auroc,
                "FINAL_test_avg_prec": test_avg_prec,
                "FINAL_val_loss": val_loss[-1][1] if cross_validate else -1,
                }
        
        logger.add_hparams(hparams_dct, results_dct) 
    return model, (preds, labs), (test_auroc, test_avg_prec), train_loss, val_loss 

def _train_step(model, dataloader, optimizer, device, epoch, logger=None):
    """Computes a single training step for the model
    """
    losses = list()

    model.train()
    
    _preds, _labels = list(), list()
    for data in dataloader:
        data.to(device)
        preds = model(data.x, data.edge_index, data.batch)
        labels = data.y[:,None].to(torch.float)
        
        loss = model.get_loss_function(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # TODO: This is ugly, fix. 
        _preds.extend(preds.cpu().detach().numpy().flatten())
        _labels.extend(data.y.cpu().detach().numpy().flatten())
        
        loss_val = loss.item()
        losses.append(loss_val)
        if logger is not None: logger.add_scalar("Loss/train", loss_val, epoch + 1)
    
    if logger is not None:
        _preds = np.round(np.array(_preds))
        _labels = np.array(_labels)
        logger.add_scalar("Accuracy/train", np.count_nonzero(_preds == _labels) / len(_labels), epoch+1)

    return np.array(losses)

def _test_step(model, dataloader, device):
    """Computes a single test step for the model
    """
    predictions, labels = list(), list()
    
    with torch.no_grad():
        model.eval()

        for data in dataloader:
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = model.predict(data.x, data.edge_index, data.batch)
            predictions.append(preds)
            labels.append(data.y)
    
    predictions = torch.squeeze(torch.cat(predictions)).cpu().numpy().flatten()
    labels = torch.cat(labels).cpu().numpy().flatten()

    return predictions, labels

def _validate_step(model, dataloader, device, epoch, logger=None):
    """Computes a single validation step for the model
    """
    predictions, labels = list(), list()
    losses = list()

    with torch.no_grad():
        model.eval()

        for data in dataloader:
            data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            preds = model.predict(data.x, data.edge_index, data.batch)
            predictions.append(preds)
            labels.append(data.y)
            
            loss = model.get_loss_function(logits, data.y[:,None].to(torch.float)) 
            loss_val = loss.item()
            losses.append(loss_val)
            if logger is not None: logger.add_scalar("Loss/val", loss_val, epoch + 1)


    predictions = torch.squeeze(torch.cat(predictions)).cpu().numpy().flatten()
    labels = torch.cat(labels).cpu().numpy().flatten()

    if logger is not None:
        _preds = np.round(predictions)
        logger.add_scalar("Accuracy/val", np.count_nonzero(_preds == labels) / len(labels), epoch+1)

    return predictions, labels, np.mean(losses)
