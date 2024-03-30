import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import config
from src.dataset import FloodDataprocessor, FloodDataset, split_train_val
from src.training_functions import Trainer, predict
from src.submission import from_full_images_array_list_to_xarray, from_results_tensor_to_list_full_image, from_xarray_to_vector
if __name__ == "__main__":

    #### PREPARE DATA ####

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['Device'] = device

    print(f'Flood processor launched')
    flood_processor = FloodDataprocessor(config)
    print(f"Flood processor Done.")
    flood_train_dataset = FloodDataset(config, flood_processor, "train")
    train_loader, val_loader = split_train_val(flood_train_dataset, config, seed=0)
    print('Set Loader and start training.')
    config['NbFeatures'] = next(iter(train_loader))[0].shape[1]
    print('Number of features: ', config['NbFeatures'])
    trainer = Trainer(config, train_loader, val_loader)
    ckpt_path = config['DataDir']+  '/' + f'checkpoint_{config["SizePatch"]}.pth'

    #### TRAINING ####
    loss, model = trainer.train_and_evaluate()
    torch.save(model.state_dict(), ckpt_path)

    ### SUBMISSION ###
    print('Start submission.')
    model = trainer.build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    flood_test_dataset = FloodDataset(config, flood_processor, "test")
    test_loader = DataLoader(flood_test_dataset, 1, shuffle=False)
    y_pred,y_prob = predict(test_loader, model, 0.05, flood_test_dataset.patches.shape[0], config['Device'], config['PastData'])
    list_image = from_results_tensor_to_list_full_image(y_prob, flood_processor.patches, flood_processor.mask, config['SizePatch'])
    x = flood_processor.labels.x
    y = flood_processor.labels.y
    xarray_score = from_full_images_array_list_to_xarray(list_image, flood_processor.test_times, config['DataDir'], x, y)
    vector = from_xarray_to_vector(xarray_score)
    submission_name = f"Test_{config['SizePatch']}"
    name_submission = config['DataDir']+ "/" + submission_name + ".csv"
    pd.DataFrame(vector).to_csv(name_submission)
    shutil.make_archive(name_submission, 'zip', config['DataDir'], submission_name + ".csv")











    