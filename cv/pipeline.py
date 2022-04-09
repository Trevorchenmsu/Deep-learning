# citation: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import util
from dataset import load_train_set, load_test_set
from model import VGG16, CustomizedModel
import argparse

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn


classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def sample_display(train_loader, batch_size):
    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    util.imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


def train(train_loader, model, criterion, optimizer, epochs):
    # Train the network. We simply have to loop over our data iterator,
    # and feed the inputs to the network and optimize.

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    return model


# def validate():


def test(test_load, model, PATH):
    data_iter = iter(test_load)
    images, labels = data_iter.next()

    # print images
    util.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    model.load_state_dict(torch.load(PATH))
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    # look at how the network performs on the whole dataset.
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_load:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



def predict(test_loader, model):
    # what are the classes that performed well, and the classes that did not perform well:
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def main():
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Current device is ", device)

    # arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str,
                        help="select: vgg, customized, pre-trained-vgg")
    parser.add_argument('--mode', type=str,
                        help="select: train, extra test")
    parser.add_argument('--data_dir', type=str,
                        help="set the dataset directory")
    parser.add_argument('--model_dir', type=str,
                        help="set directory to save the model ")

    args = parser.parse_args()

    num_workers = 4
    batch_size = 4
    epochs = 30
    model = None

    if args.mode == 'train':
        train_loader = load_train_set(args.data_dir, args.batch_size, args.num_workers)

        if args.model == 'vgg':
            model = VGG16()

        if args.mode == "customized":
            model = CustomizedModel()

        # if args,mode == "pre-trained-vgg":
        #     load model

        model.to(device)
        train_loader.to(device)

        # Define a Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model = train(train_loader, model, criterion, optimizer, args.epoch)

        # save  trained model
        PATH = args.model_dir + 'SVHN_' + args.model + '.pth'
        torch.save(model.state_dict(), PATH)

    if args.mode == 'test':
        test_loader = load_test_set(args.data_dir, args.batch_size, args.num_workers)

        if args.model == 'vgg':
            model = VGG16()

        if args.mode == "customized":
            model = CustomizedModel()

        # if args,mode == "pre-trained-vgg":
        #     load model
        PATH = args.model_dir + 'SVHN_' + args.model + '.pth'
        test(test_loader, model, PATH)


if __name__ == '__main__':
    main()








