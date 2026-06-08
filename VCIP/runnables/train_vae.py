import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import time
import argparse
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger
import omegaconf
from pytorch_lightning.loggers import MLFlowLogger
import importlib
import glob

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
from hydra.utils import get_original_cwd
import pickle
from geomloss import SamplesLoss

from src.utils.utils import set_seed
from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, to_double
from src.utils.helper_functions import ExperimentManager
from pathlib import Path
import hashlib

import warnings
from src.utils.helper_functions import write_csv, check_csv
warnings.filterwarnings("ignore")

os.environ['HYDRA_FULL_ERROR'] = '1'

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

@hydra.main(config_name='config.yaml', config_path='../configs/')
def main(args: DictConfig):
    # Basic setup
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    set_seed(args.exp.seed)

    # Check optimization requirements
    current_dir = get_absolute_path('./')
    csv_dir = current_dir + f'/{args.exp.test}'
    optimize_interventions = True
    if not check_csv(csv_dir, args):
        optimize_interventions = False

    # Data loading and processing
    original_cwd = get_original_cwd()
    args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
    
    path = os.path.join(args['exp']['processed_data_dir'], f"seed_{args.exp.seed}.pkl")
    if True:
        os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_multi()
        dataset_collection = to_float(dataset_collection)
        if args['dataset']['static_size'] > 0:
            dims = len(dataset_collection.train_f.data['static_features'].shape)
            if dims == 2:
                dataset_collection = repeat_static(dataset_collection)
    else:
        with open(path, 'rb') as file:
            dataset_collection = pickle.load(file)
            if args['dataset']['static_size'] > 0:
                dims = len(dataset_collection.train_f.data['static_features'].shape)
                if dims == 2:
                    dataset_collection = repeat_static(dataset_collection)

    # Model initialization and training
    module_path, class_name = args["model"]["_target_"].rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(args, dataset_collection)

    # Setup model directory with seed subdirectory
    current_dir = get_absolute_path('./')
    model_dir = os.path.join(current_dir, 'models', f'{args.exp.seed}')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model.ckpt')
    
    if os.path.exists(model_path):
        # Load existing model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded existing model from {model_path}")
    else:
        # Train new model
        logger_board = TensorBoardLogger(save_dir=model_dir, name='', version='')
        
        trainer = pl.Trainer(
            logger=logger_board,
            max_epochs=args.exp.epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            devices=args.exp.gpus,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=model_dir,
                    monitor='val_loss',
                    filename='model',
                    save_top_k=1,
                    mode='min'
                )
            ]
        )
        
        trainer.fit(model)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Trained and saved new model to {model_path}")

    # Optimization phase
    if args.exp.rank:
        if args.dataset.name in ('mimic3_real', 'mimic4_mortality') or \
                (args.dataset.name == 'tumor_generator' and args.dataset.coeff in [1,2,3,4]):
            model.optimize_interventions_discrete()
    else:
        if not optimize_interventions:
            return
            
        num_iterations = 100
        result = model.optimize_interventions(num_iterations=num_iterations, learning_rate=0.01, batch_size=128)
        
        # Save results
        result_path = os.path.join(model_dir, "results.txt")
        with open(result_path, 'a') as f:
            f.write(result)
        
        write_csv(result, csv_dir, args)
        return result

if __name__ == "__main__":
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    warnings.filterwarnings("ignore")
    os.environ['HYDRA_FULL_ERROR'] = '1'
    OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)
    main()