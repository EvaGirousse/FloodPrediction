import time
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import numpy as np

class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config['Device']
    
    def build_model(self):
        model = self.config['Model']({}, self.config).to(self.device)
        return model


    def evaluate(self, loader,model,prob_treshold, return_pred=False, return_all=False, shuffle_dim=None):
        y_true = []
        y_pred = []
        y_prob = []
        all_X = []
        all_true = []
        all_prob = []
        model.eval()
        with torch.inference_mode():
            for batch_X,batch_y in loader:
                if shuffle_dim is not None:
                    batch_X[:, shuffle_dim] = batch_X[torch.randperm(len(batch_X)), shuffle_dim]
                batch_y_pred = model(batch_X)
                batch_prob = nn.Sigmoid()(batch_y_pred).squeeze().cpu()
                all_X.append(batch_X)
                all_prob.append(batch_prob)
                all_true.append(batch_y)
                for i in range(len(batch_X)):
                    y_true.append(batch_y[i].cpu().numpy())
                    y_prob.append(batch_prob[i].cpu().numpy())
                    y_temp = batch_prob[i].clone()
                    y_temp[y_temp<prob_treshold] = 0
                    y_temp[y_temp>=prob_treshold] = 1
                    y_pred.append(y_temp.cpu().numpy())
        all_X = np.concatenate(all_X, axis = 0)
        all_prob = np.concatenate(all_prob, axis = 0)
        all_true = np.concatenate(all_true, axis = 0)
        y_true = np.array(y_true).reshape(-1,)
        mask = np.where(y_true == -1)
        y_true = np.delete(y_true, mask)
        y_prob = np.delete(np.array(y_prob).reshape(-1,),mask)
        y_pred = np.delete(np.array(y_pred).reshape(-1,),mask)
        auc = roc_auc_score(y_true,y_prob)
        print(f"Losses | AUC = {auc}, shape: {y_pred.shape}, nb of true flood events: {y_true.sum()}, proportion: {y_true.sum()/y_true.shape[0]}, mean prob: {y_prob.mean()}, max prob: {y_prob.max()}")
        if return_pred:
            return y_true, y_prob, y_pred
        elif return_all:
            return all_X, all_prob, all_true
        else:
            return  auc       
        
    def train_and_evaluate(self): 
        model =  self.build_model()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        n_epochs = self.config['NumEpochs']
        for epoch in range(n_epochs):
            epoch_loss = train_one_epoch(self.train_loader,model = model,optimizer=optimizer,loss_fn = loss_fn)
            print(f"{epoch}: {epoch_loss}")
        auc = self.evaluate(self.val_loader,model=model,prob_treshold=0.05)
        return auc, model
    
    
    

def predict(loader,model,prob_treshold, nb_patches, device, past_data=False):

    y_pred = []
    y_prob = []
    model.eval()
    patch_nb = 0
    with torch.inference_mode():
        start_time = time.time()
        succeeded = True
        for batch_X in loader:
            if past_data:
                if patch_nb<nb_patches:
                    previous_pred = torch.zeros(1, 1, batch_X.shape[2], batch_X.shape[3])
                else:
                    previous_pred = y_pred[patch_nb - nb_patches].unsqueeze(0).unsqueeze(0)
                batch_X = torch.concat([batch_X, previous_pred.to(device)], dim=1)
            batch_y_pred = model(batch_X)
            batch_prob = nn.Sigmoid()(batch_y_pred).squeeze(1).cpu()
            for i in range(len(batch_X)):
                y_prob.append(batch_prob[i].cpu())
                y_temp = batch_prob[i].clone()
                y_temp[y_temp<prob_treshold] = 0
                y_temp[y_temp>=prob_treshold] = 1
                y_pred.append(y_temp.cpu())
            patch_nb +=1
            duration = time.time() - start_time
            if duration > 3600:
                succeeded = False
                break
    if not succeeded:
        return None, None
    else:
        return  y_pred,y_prob

def train_one_epoch(loader, model,optimizer,loss_fn):
    running_loss = 0.
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = torch.where(labels!=-1, loss_fn(outputs, labels), torch.zeros_like(labels)).sum() / (labels != -1).sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)