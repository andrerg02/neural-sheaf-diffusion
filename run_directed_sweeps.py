import wandb
import argparse
import json 
import os 
import itertools 

# Parse the arguments
def get_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BundleSheaf')#, required=True)
    parser.add_argument('--project', type=str, default='csn_chamsquir_sweeps')
    parser.add_argument('--datasets', type=str, default='squirrel')
    return parser

def get_sweep_filename(dataset): 
    return f'sweeps_config/model_sweep_id_{dataset}.json'

def get_model_list(args): 
    # Check if multiple models were passed as arguments 
    if ',' not in args.model: 
        return [args.model] 
    else: 
        return [model for model in args.model.split(',')]   
    
def create_sweep_config(args, dataset): 
    # Define the sweep configuration
    sweep_config = {
        'program': 'exp/run.py',
        'name': f'{args.model}_{dataset}',
        'method': 'grid', # we could also use random
        'metric': {
            'goal': 'maximize',
            'name': 'val_acc'
        },
        'parameters': {
            'dataset': {
                'values': [dataset] 
            },
            'd': {
                'values': [2,3,4,5]
            },
            'hidden_channels': {
                'values': [32,64]
            },
            'layers': {
                'values': [2,3,4,5]
            },
            'left_weights': {
                'values': [True, False]
            },
            'right_weights': {
                'values': [True, False]
            },
            'epochs': {
                'value': 2500
            },
            'early_stopping': {
                'value': 500
            },
            'lr': {
                'value': 0.02
            },
            'weight_decay': {
                'value': 0.0
            },
            'input_dropout': {
                'value': 0.5
            },
            'dropout': {
                'value': 0.5
            },
            'orth': {
                'value': 'householder'
            },
            'folds': {
                'value': 10
            },
            'use_act': {
                'value': True
            },
            'model': {
                'value': args.model 
            },
            'edge_weights': {
                'value': False
            },
            'entity': {
                'value': 'andrerg00'
            }
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity='andrerg00')
    print(f'Sweep ID for {args.model}: {sweep_id}')
    return sweep_id 

if __name__ == '__main__': 
    parser = get_parser() 
    args = parser.parse_args()

    model_lst = get_model_list(args)
    datasets = args.datasets.split(',') if ',' in args.datasets else [args.datasets]
    for dataset in datasets: 
        filename = get_sweep_filename(dataset) 
        sweep_dct = dict() 
        if not os.path.exists('sweeps_config'): 
            os.mkdir('sweeps_config') 
        if os.path.exists(filename): 
            sweep_dct = json.load(open(filename)) 
        for model in model_lst: 
            args.model = model 
            sweep_id = create_sweep_config(args, dataset) 
            sweep_dct[model] = sweep_id    
        json.dump(
            sweep_dct, open(filename, 'w')
        )