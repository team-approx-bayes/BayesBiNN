import os
import json



def save_train_history_CL(args,test_acc):
    dict_save_path = os.path.join(args.out_dir, 'dicts', 'train_hist_{}.json'.format(args.experiment_id))
    os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
    with open(dict_save_path, 'w') as f:
        json.dump({'test_acc':  test_acc}, f)

def save_train_history(args, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    dict_save_path = os.path.join(args.out_dir, 'dicts', 'train_hist_{}.json'.format(args.experiment_id))
    os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
    with open(dict_save_path, 'w') as f:
        json.dump({'train_loss': train_loss, 'train_acc': train_acc,
                     'val_loss':   val_loss,   'val_acc':   val_acc,
                    'test_loss':  test_loss,  'test_acc':  test_acc}, f)
