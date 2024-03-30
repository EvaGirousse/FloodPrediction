import torch
import numpy as np
from torch.utils.data import Dataset

from src.inr_networks import ModulatedSiren
from src.metalearning.metalearning import outer_step

class DatasetSamples(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """
        self.v = v
        self.z = torch.zeros((v.shape[0], latent_dim))
        self.c = grid

        #if sample_ratio_batch == None:
        #    self.n_points = None
        #else:
        #    self.n_points = int(sample_ratio_batch * self.c.shape[1])

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx, full_length=False):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if self.n_points == None or full_length == True:    
        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]
        
        #else:
        #    permutation = torch.randperm(self.c.shape[1])[:self.n_points]
        #    sample_v = self.v[idx, permutation, ...]
        #    sample_z = self.z[idx, ...]
        #    sample_c = self.c[idx, permutation, ...]


        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values

def load_inr_weights(config, variable, device="cpu"):
    device = torch.device(device)
    save_dir = config['DataDir']
    latent_dim = config['INRLatentDim']
    epochs = config['INREpochs']
    inr_training_input = torch.load(f"{save_dir}/models_{variable}_{latent_dim}_{epochs}.pt",map_location=device)
    output_dim = 1
    input_dim = 2       
    hidden_dim=128
    depth=6
    inr = ModulatedSiren(
                    dim_in=input_dim,
                    dim_hidden=hidden_dim,
                    dim_out=output_dim,
                    num_layers=depth,
                    w0=30.0,
                    w0_initial=30.0,
                    use_bias=True,
                    modulate_scale=False,
                    modulate_shift=True,
                    use_latent=True,
                    latent_dim=latent_dim,
                    modulation_net_dim_hidden=64,
                    modulation_net_num_layers=1,
                    mu=0,
                    sigma=1,
                    last_activation=None,
                    )
    inr.load_state_dict(inr_training_input["inr"])
    inr = inr.to(device)
    return inr

def interpolate_and_save(inr, ref_data, ref_grid, grid_patches, size_patch, device="cpu"):
    """
    """
    device = torch.device(device)
    test_inner_steps=3
    loss_type="mse"
    use_rel_loss=True
    alpha = 0.01
    step_show=100
    mean = ref_data.mean().values.item()
    std = ref_data.std().values.item()
    full_preds = []
    for date in ref_data.time:
        ref_map = np.expand_dims(ref_data.sel(time=date).values,0)
        ref_map = (ref_map - mean)/std
        trainset = DatasetSamples(ref_map, ref_grid, 128)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

        # We train our vector Z on the considered day
        for substep, (w_map_test, modulations_test, coords, idx) in enumerate(train_loader):
            inr.eval()
            w_map_test = w_map_test.to(torch.float32).to(device)
            modulations_test = modulations_test.to(device)
            coords = coords.to(torch.float32).to(device)

            outputs_test = outer_step(
                        inr.to(device),
                        coords.to(device),
                        w_map_test.to(device),
                        test_inner_steps,
                        alpha,
                        is_train=False,
                        return_reconstructions=step_show, # type: ignore 
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type=loss_type,
                        modulations=torch.zeros_like(modulations_test).to(device), # type: ignore
                        )
        # We predict the value for all the patches from this day
        preds = []
        with torch.no_grad():
            size_batch = grid_patches.shape[0]//10
            for i in range(0, grid_patches.shape[0], size_batch):
                x = grid_patches[i:min(i+size_batch, grid_patches.shape[0])]
                x = torch.tensor(x).float()
                predicted_map = inr.to(device).modulated_forward(x.to(device), outputs_test['modulations'].to(device))
                preds.append(predicted_map)
        preds = torch.vstack(preds).reshape(-1,size_patch,size_patch,1).cpu()
        full_preds.append(preds)
    return np.asarray(full_preds)
    