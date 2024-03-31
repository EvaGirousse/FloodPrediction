from src.models import BabyCNN

config = {
    "DataDir": "data",
    "PreprocessedDir": "data",
    "TargetsFile": "final_label_Full_Rez.nc", 
    "Features": 16,
    "PastData": False,
    "BatchSize": 1024,
    "SizePatch": 4,
    "MinLon": -5.0,
    "MaxLon": 10.0,
    "MinLat": 41.0,
    "MaxLat": 51.0,
    "t2mFile": "ERA5_train.nc",
    "tpFile": "precipitations_2000_2004_latlon.nc",
    'INRLatentDim': 128,
    'INREpochs': 1500,
    'Dates': {
        'StartTrain': '2002-08-04',
        'EndTrain': '2003-03-09',
        'StartTest': '2003-11-02',
        'EndTest': '2003-12-28'
    },
    'NumEpochs': 30,
    'Model': BabyCNN
}
