import torch
import torchvision
import torchvision.transforms as transforms

# cited: https://pytorch.org/vision/main/generated/torchvision.datasets.SVHN.html

def load_train_set(root, batch_size, num_workers):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose([
                                    transforms.CenterCrop((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])


    train_set = torchvision.datasets.SVHN(root=root,
                                          split='train',
                                          download=False,
                                          transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    return train_loader

def load_extra_set(root, batch_size, num_workers):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose([
                                    transforms.CenterCrop((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])


    extra_set = torchvision.datasets.SVHN(root=root,
                                          split='extra',
                                          download=False,
                                          transform=transform)

    extra_loader = torch.utils.data.DataLoader(extra_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    return extra_loader

def load_test_set(root, batch_size, num_workers):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose([
                                    transforms.CenterCrop((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])

    test_set = torchvision.datasets.SVHN(root=root,
                                         split='test',
                                         download=False,
                                         transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    return test_loader