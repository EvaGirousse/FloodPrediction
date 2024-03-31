import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from src.inr_utils import load_inr_weights, interpolate_and_save
from src.utils import get_label_patches, get_useful_patches_index, patchify, process_image, get_static_patches
import pandas as pd
import xarray as xr
from torch.utils.data import SubsetRandomSampler, DataLoader
from datetime import timedelta
import time


class FloodDataprocessor:
    def __init__(self, config, save=None):
        self.save_path = config['DataDir']
        self.process_dir = config['PreprocessedDir']+f"/patch_{config['SizePatch']}/"
        if save is None:
            save = not os.path.isdir(self.process_dir)
        if save:
            os.makedirs(self.process_dir, exist_ok=True)
        self.train_times = pd.date_range(config['Dates']['StartTrain'], config['Dates']['EndTrain'], freq='7D')
        self.test_times = pd.date_range(config['Dates']['StartTest'], config['Dates']['EndTest'], freq='7D')
        if "TargetsFile" in config:
            self.labels = xr.open_dataset(self.save_path+f"/{config['TargetsFile']}", engine="scipy")
        # Load raw input data
        self.mask = xr.open_dataset(self.save_path+"/relevent_data_tag_binary_mask.nc")    
        print(f'Data Flood with save = {save}')  
        if save:
            # Process data
            self.static_data = xr.open_dataset(self.save_path+'/static_Full_Rez.nc')
            self.static_data_mean = self.static_data.mean()
            self.static_data_std = self.static_data.std()
            self.static_data = (self.static_data - self.static_data.mean())/self.static_data.std()
            self.patches = get_useful_patches_index(self.labels, config)
            self.precipitations_train, self.precipitations_test = self.get_inr_interpolation(config, 'tp')
            self.temperature_train, self.temperature_test = self.get_inr_interpolation(config, 't2m')
            self.static_data = get_static_patches(self.static_data, self.patches, config['SizePatch'])
            self.train_labels = self.labels.sel(time=slice(self.train_times[0], self.train_times[-1]))
            self.train_labels = get_label_patches(self.train_labels,self.patches, config['SizePatch'])
            with open(f"{self.process_dir}/patches_index.pkl", "wb") as f:
                pickle.dump(self.patches, f)
            with open(f"{self.process_dir}/precipitations_train.pkl", "wb") as f:
                pickle.dump(self.precipitations_train, f)
            with open(f"{self.process_dir}/precipitations_test.pkl", "wb") as f:
                pickle.dump(self.precipitations_test, f)
            with open(f"{self.process_dir}/temperature_train.pkl", "wb") as f:
                pickle.dump(self.temperature_train, f)
            with open(f"{self.process_dir}/temperature_test.pkl", "wb") as f:
                pickle.dump(self.temperature_test, f)
            with open(f"{self.process_dir}/static_patches.pkl", "wb") as f:
                pickle.dump(self.static_data, f)
            with open(f"{self.process_dir}/train_labels.pkl", "wb") as f:
                pickle.dump(self.train_labels, f)
        else:
            while len(os.listdir(self.process_dir)) < 7:
                print(f"Waiting for {self.process_dir} to be completed: {len(os.listdir(self.process_dir))} < 7")
                time.sleep(60)
            loaded = False
            while not loaded: 
                try:
                    with open(f"{self.process_dir}/patches_index.pkl", "rb") as f:
                        self.patches = pickle.load(f)
                    with open(f"{self.process_dir}/precipitations_train.pkl", "rb") as f:
                        self.precipitations_train = pickle.load(f)
                    with open(f"{self.process_dir}/precipitations_test.pkl", "rb") as f:
                        self.precipitations_test = pickle.load(f)
                    with open(f"{self.process_dir}/temperature_train.pkl", "rb") as f:
                        self.temperature_train = pickle.load(f)
                    with open(f"{self.process_dir}/temperature_test.pkl", "rb") as f:
                        self.temperature_test = pickle.load(f)
                    with open(f"{self.process_dir}/static_patches.pkl", "rb") as f:
                        self.static_data = pickle.load(f)
                    with open(f"{self.process_dir}/train_labels.pkl", "rb") as f:
                        self.train_labels = pickle.load(f)
                    loaded = True
                except Exception as e:
                    print(f"Loading data failed with {e}")


    def get_inr_interpolation(self, config, variable):
        inr = load_inr_weights(config, variable, config['Device'])
        xr_ref = xr.open_dataset(self.save_path+"/"+config[f'{variable}File'])
        longitude = xr_ref.longitude
        latitude = xr_ref.latitude
        xr_ref = xr_ref[variable]
        grid_ref= torch.stack(torch.meshgrid((torch.tensor(longitude.to_numpy()) - config['MinLon'])/(config['MaxLon']-config['MinLon']), ((torch.tensor(latitude.to_numpy())-config['MinLat'])/(config['MaxLat']-config['MinLat'])), indexing="xy"), dim=-1).unsqueeze(0)
        labels_grid = torch.stack(torch.meshgrid(torch.tensor(self.labels.x.to_numpy()), torch.tensor(self.labels.y.to_numpy()), indexing="xy"), dim=-1)
        labels_grid = process_image(labels_grid, config['SizePatch'])
        patches_coord, index_coord = patchify(labels_grid, config['SizePatch'])
        final_labels_patches = []
        for i, idx in enumerate(index_coord):
            if i in self.patches :
                final_labels_patches.append(patches_coord[i])
        final_labels_patches = np.asarray(final_labels_patches)
        final_labels_patches[:,:,:,0] = (final_labels_patches[:,:,:,0]-config['MinLon'])/(config['MaxLon']-config['MinLon'])
        final_labels_patches[:,:,:,1] = (final_labels_patches[:,:,:,1]-config['MinLat'])/(config['MaxLat']-config['MinLat'])
        if pd.to_datetime(xr_ref.time[0].values).hour>0:
            last_day = 1
        else:
            last_day = 0
        train_ref = xr_ref.sel(time=slice(str(self.train_times[0]-timedelta(days=6)), str(self.train_times[-1]+timedelta(days=last_day))))
        test_ref = xr_ref.sel(time=slice(str(self.test_times[0]-timedelta(days=6)), str(self.test_times[-1]+timedelta(days=last_day))))
        train_patches = interpolate_and_save(inr, train_ref, grid_ref, final_labels_patches, config['SizePatch'], device=config['Device'])
        test_patches = interpolate_and_save(inr, test_ref, grid_ref, final_labels_patches,  config['SizePatch'], device=config['Device'])
        train_patches = train_patches.reshape(7, train_patches.shape[0]//7, train_patches.shape[1], train_patches.shape[2], train_patches.shape[3], train_patches.shape[4])
        test_patches = test_patches.reshape(7, test_patches.shape[0]//7, test_patches.shape[1], test_patches.shape[2], test_patches.shape[3], test_patches.shape[4])
        if variable == "tp":
            train_patches = np.max(train_patches, axis=0)
            test_patches = np.max(test_patches, axis=0)
        if variable == "t2m":
            train_patches = np.mean(train_patches, axis=0)
            test_patches = np.mean(test_patches, axis=0)
        return train_patches, test_patches
    

class FloodDataset(Dataset):
    def __init__(self, config, data_processor, label, features=None):        
        self.static_data = data_processor.static_data
        self.patches = data_processor.patches
        self.config = config
        if label=="train":
            self.times = data_processor.train_times
            self.labels = data_processor.train_labels
            self.precipitations = data_processor.precipitations_train
            self.temperature = data_processor.temperature_train
        else:
            self.times = data_processor.test_times
            self.precipitations = data_processor.precipitations_test
            self.temperature = data_processor.temperature_test
        self.label = label
        self.size_patch = config['SizePatch']
        self.device = config['Device']
        if features is not None:
            self.static_data = self.static_data[np.array(features).astype(bool)]
        
    def __len__(self):
        nb_weeks = self.times.shape[0]
        nb_patches = self.patches.shape[0]
        if (self.label == "train") and self.config['PastData']:
            return (nb_weeks-1)*nb_patches
        else:
            return nb_weeks * nb_patches
    
    def __getitem__(self, idx):
        if self.label == "train":
            if self.config['PastData']:
                idx += self.patches.shape[0]
            week = idx // self.patches.shape[0]
            id_patch = idx - week*self.patches.shape[0]
            X = np.concatenate([self.static_data[:,id_patch],self.precipitations[week,id_patch].reshape((1,self.size_patch,self.size_patch)),self.temperature[week,id_patch].reshape((1,self.size_patch,self.size_patch)),
                            ], axis = 0)
            if self.config['PastData']:
                X = np.concatenate([X, self.labels[week-1, id_patch].reshape((1, self.size_patch, self.size_patch))], axis=1)

            y = self.labels[week, id_patch]
            return torch.tensor(X).float().to(self.device), torch.tensor(y).float().to(self.device)
        else:
            week = idx // self.patches.shape[0]
            id_patch = idx - week*self.patches.shape[0]
            X = np.concatenate([self.static_data[:,id_patch],self.precipitations[week,id_patch].reshape((1,self.size_patch,self.size_patch)),
                                self.temperature[week,id_patch].reshape((1,self.size_patch,self.size_patch))], axis = 0)
            return torch.tensor(X).float().to(self.device)
        
def split_train_val(train_data, config, seed, size=0.2):
    split = int(np.floor(size * train_data.__len__()))
    np.random.seed(seed)
    indices = list(range(train_data.__len__()))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_data, batch_size=config['BatchSize'], sampler=train_sampler)
    validation_loader = DataLoader(train_data, batch_size=config['BatchSize'], sampler=valid_sampler)
    return train_loader, validation_loader


        
    







