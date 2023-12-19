import csv, os, pickle
import numpy as np
import torch

def stratified_kfold_split(labels_fpath, k, include_validation = False):
    """Computes stratified K-fold CV split (as generator)
    """
    positives, negatives = _load_samples_split(labels_fpath)

    positives = np.array_split(np.random.permutation(positives), k)
    negatives = np.array_split(np.random.permutation(negatives), k)
    
    for ki in range(k):
        test_set = [*positives[ki], *negatives[ki]]

        if include_validation:
            val_idx = (ki + 1) % k
            val_set = [*positives[val_idx], *negatives[val_idx]]

            train_set = list()
            for kj in filter(lambda kj: kj != ki and kj != val_idx, range(k)):
                train_set.extend([*positives[kj], *negatives[kj]])

            yield train_set, val_set, test_set
        else:
            train_set = list()
            for kj in filter(lambda kj: kj != ki, range(k)):
                train_set.extend([*positives[kj], *negatives[kj]])
            
            yield train_set, test_set

def repeated_stratified_kfold_splits(labels_fpath, k, r, include_validation = False):
    """Computes repeated leave a fold out CV (as generator)
    """
    positives, negatives = _load_samples_split(labels_fpath)
    pos_fold_size = len(positives) // k
    neg_fold_size = len(negatives) // k

    for ni in range(r):
        test_set = [*np.random.choice(positives, pos_fold_size), *np.random.choice(negatives, neg_fold_size)]
        
        if include_validation:
            _positives = [pi for pi in positives if pi not in test_set]
            _negatives = [ni for ni in negatives if ni not in test_set]
            val_set = [*np.random.choice(_positives, pos_fold_size), *np.random.choice(_negatives, neg_fold_size)]
            
            _positives = [pi for pi in _positives if pi not in val_set]
            _negatives = [ni for ni in _negatives if ni not in val_set]
            
            yield [*_positives, *_negatives], val_set, test_set
        else:
            _positives = [pi for pi in positives if pi not in test_set]
            _negatives = [ni for ni in negatives if ni not in test_set]

            yield [*_positives, *_negatives], test_set

def make_inner_kfold_split(data_split, excluded_index: int):
    train_folds = [ds[-1] for i, ds in enumerate(data_split) if i != excluded_index]
    test_fold = data_split[excluded_index][-1]

    inner_data_split = list()
    for i in range(len(data_split) - 1):
        val_fold = train_folds[i]
        train_fold = [sample_id for j, samples in enumerate(train_folds) for sample_id in samples if j != i]

        inner_data_split.append((train_fold, val_fold, test_fold))
        assert all(ti not in test_fold for ti in train_fold)
        assert all(vi not in test_fold for vi in val_fold)

    return inner_data_split, train_folds

def get_all_sample_ids(labels_fpath):
    """Returns a list of all sample IDs from the dataset.
    """
    positives, negatives = _load_samples_split(labels_fpath)
    all_samples = positives + negatives
    return all_samples

def _load_samples_split(labels_fpath):
    """Returns two lists of sample ids split by positive and negative sample
    """
    with open(labels_fpath, 'r') as ifile:
        reader = csv.reader(ifile, delimiter=',')
        next(reader)
        
        positives, negatives = list(), list()
        for row in reader:
            if int(row[1]) != 0:
                positives.append(row[0])
            else:
                negatives.append(row[0])
    
    return positives, negatives

_model_ext = '_model_statedict.pt'
_kwargs_ext = '_model_init.p'
def save_model(model, kwargs, fpath_rt, model_ext: str = _model_ext, kwargs_ext: str = _kwargs_ext):
    model_fpath = fpath_rt + model_ext 
    torch.save(model.state_dict(), model_fpath)
    
    kwargs_fpath = fpath_rt + kwargs_ext 
    with open(kwargs_fpath, 'wb+') as f:
        pickle.dump(kwargs, f)

def load_model(model_type, fpath_rt, model_ext: str = _model_ext, kwargs_ext: str = _kwargs_ext):
    model_fpath = fpath_rt + model_ext 
    kwargs_fpath = fpath_rt + kwargs_ext

    if not os.path.exists(model_fpath) or not os.path.exists(kwargs_fpath):
        raise ValueError("Could not find model or kwargs file in given location.")
    
    with open(kwargs_fpath, 'rb') as f:
        kwargs = pickle.load(f)
    
    model = model_type(**kwargs)
    
    state_dict = torch.load(model_fpath)
    model.load_state_dict(state_dict)
    
    model.eval() # TODO: Is this necessary? Doesn't hurt, but I think this is overthinking the function.

    return model
