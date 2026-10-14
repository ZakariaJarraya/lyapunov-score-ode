from .resnet import resnet18, resnet50, WideResNet_70_16
import torch.nn as nn
#from .vit import vit_b16  


def get_classifier(name):

    if name == "resnet18":
        return resnet18()
    
    if name == "resnet50":
        return resnet50()
    
    if name == "wideresnet":
        return WideResNet_70_16()

    #if name == "vit":
        #return vit_b16()

    raise ValueError("Unknown classifier")




def get_dataset_info(name):
    name = name.lower()
    
    # Metadata for non-MedMNIST datasets
    standard_info = {
        'cifar10': {'num_classes': 10, 'img_size': 32},
        'mnist':   {'num_classes': 10, 'img_size': 32}, # We resized to 32 in get_dataset
        'imagenet': {'num_classes': 1000, 'img_size': 224},
        'tiny_imagenet': {'num_classes': 200,  'img_size': 224}
    }

    if name in standard_info:
        return standard_info[name]
    
    # Handle MedMNIST
    elif name in ['pneumoniamnist', 'dermamnist']:
        try:
            from medmnist import INFO
            info = INFO[name]
            return {
                'num_classes': len(info['label']),
                'img_size': 224 # Based on the size=224 passed in get_dataset
            }
        except ImportError:
            raise ImportError("medmnist package is required for this dataset.")
            
    else:
        raise ValueError(f"Metadata not found for dataset: {name}")


def adapt_model(model, dataset_name):
    """
    Adjusts ResNet stem and head based on dataset characteristics.
    """
    info = get_dataset_info(dataset_name)
    num_classes = info["num_classes"]
    img_size = info["img_size"]
    if type(model).__name__.lower() != "wideresnet":
        # ---- Stem Modification ----
        # If the image is small (CIFAR/MNIST), use a smaller kernel and remove maxpool
        # to preserve spatial resolution in the early layers.
        if img_size <= 32:
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()
        else:
            # Standard ResNet stem for 224x224
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Head Modification ----
        # Adjust the final fully connected layer to match the dataset class count
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
