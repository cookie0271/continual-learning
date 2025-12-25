import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

#----------------------------------------------------------------------------------------------------------#

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-context experiment.'''

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[name]
    config['output_units'] = config['classes']
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config

#----------------------------------------------------------------------------------------------------------#

def get_context_set(name, scenario, contexts, data_dir="./datasets", only_config=False, verbose=False,
                    exception=False, normalize=False, augment=False, singlehead=False,
                    train_set_per_class=False, adversarial_label_shuffle=False):
    '''Load, organize and return a context set (both train- and test-data) for the requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first context (permMNIST) or digits
                            are not shuffled before being distributed over the contexts (e.g., splitMNIST, CIFAR100)
    [adversarial_label_shuffle]:
                    <bool>; if True, shuffle labels within each context (tasks 2..T) to create adversarial label
                            mappings while keeping them consistent between train and test datasets'''

    ## NOTE: options 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # Define data-type
    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST currently has no support for separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    if adversarial_label_shuffle and name != "CIFAR100":
        raise ValueError("Adversarial label shuffle is currently only supported for 'CIFAR100'.")
    if adversarial_label_shuffle and train_set_per_class:
        raise NotImplementedError('Adversarial label shuffle is not supported with train_set_per_class=True.')

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["CIFAR100_denorm"]
    config['adversarial_label_shuffle'] = adversarial_label_shuffle
    # check for number of contexts
    if contexts > config['classes'] and not name=="permMNIST":
        raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, config['classes']))
    # -how many classes per context?
    classes_per_context = 10 if name=="permMNIST" else int(np.floor(config['classes'] / contexts))
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario=='domain' or
                                                    (scenario=="task" and singlehead)) else classes_per_context*contexts
    # -if only config-dict is needed, return it
    if only_config:
        return config

    label_permutation_summary = None
    label_shuffle_strategy = None

    # Depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x*classes_per_context
            ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
        labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
            list(np.array(range(classes_per_context)) + classes_per_context * context_id)
            for context_id in range(contexts)
        ]
        labels_per_dataset_test = [
            list(np.array(range(classes_per_context)) + classes_per_context * context_id)
            for context_id in range(contexts)
        ]

        # pre-compute label permutations per context (identity for task 1)
        label_shuffle_mappings = []
        label_permutation_summary = []
        if adversarial_label_shuffle:
            label_shuffle_strategy = (
                'reuse-previous-classes' if scenario == 'class' else 'within-context'
            )
        previous_label_pool = []
        for context_id, labels in enumerate(labels_per_dataset_test):
            labels_int = [int(l) for l in labels]
            source_pool = None
            if adversarial_label_shuffle and context_id > 0:
                if scenario == 'class':
                    source_pool = list(previous_label_pool)
                    if len(source_pool) < len(labels_int):
                        raise ValueError(
                            'Not enough previously seen classes to construct adversarial shuffle.'
                        )
                    permuted_labels = list(
                        np.random.choice(source_pool, size=len(labels_int), replace=False)
                    )
                else:
                    permuted_indices = np.random.permutation(len(labels_int))
                    permuted_labels = [int(labels_int[idx]) for idx in permuted_indices]
            else:
                permuted_labels = [int(l) for l in labels_int]
            mapping = {int(label): int(permuted_labels[idx]) for idx, label in enumerate(labels_int)}
            label_shuffle_mappings.append(mapping)
            summary_entry = {
                'context': context_id + 1,
                'original': labels_int,
                'permuted': permuted_labels,
            }
            if source_pool is not None:
                summary_entry['source_pool'] = source_pool
            label_permutation_summary.append(summary_entry)
            previous_label_pool.extend(labels_int)

        def _make_target_transform(labels, mapping, context_index):
            offset = int(labels[0])
            apply_shuffle = adversarial_label_shuffle and context_index > 0
            if scenario == 'domain' or (scenario == "task" and singlehead):
                return lambda y, mapping=mapping, offset=offset: int(mapping.get(int(y), int(y)) - offset)
            elif apply_shuffle:
                return lambda y, mapping=mapping: int(mapping.get(int(y), int(y)))
            else:
                return None

        # split the train and test datasets up into sub-datasets
        train_datasets = []
        for labels in labels_per_dataset_train:
            first_label = int(labels[0])
            context_index = int(first_label // classes_per_context) if classes_per_context > 0 else 0
            context_index = max(0, min(context_index, len(label_shuffle_mappings) - 1))
            mapping = label_shuffle_mappings[context_index]
            target_transform = _make_target_transform(labels, mapping, context_index)
            train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
        test_datasets = []
        for context_index, labels in enumerate(labels_per_dataset_test):
            mapping = label_shuffle_mappings[context_index]
            target_transform = _make_target_transform(labels, mapping, context_index)
            test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))

    config['label_permutations'] = label_permutation_summary
    config['label_shuffle_strategy'] = label_shuffle_strategy

    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets), config)