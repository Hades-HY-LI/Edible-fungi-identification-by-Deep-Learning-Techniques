#!/usr/bin/env python
# coding: utf-8

## Convoluted Neural Network

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
# plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = os.getcwd() + '\\data2'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True)
              for x in ['train', 'validation', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])   # 0: edible, 1: poisonous

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_record = []
    epoch_acc_record = []
    poison_identified_record = []
    epoch_recall_record = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            poison_identified = 0.0
            poison_sum = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                poison_identified += torch.sum((preds == labels.data) * (labels.data == 1))
                poison_sum += torch.sum(labels.data == 1)
            if phase == 'train':
                scheduler.step()

            # TODO: save the epoch loss, make figs later
            # when epoch loss stops decresing, it's time to change the learning rate scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_recall = poison_identified / poison_sum

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f}')
            epoch_loss_record.append(epoch_loss)
            epoch_acc_record.append(epoch_acc)
            poison_identified_record.append(poison_identified)
            epoch_recall_record.append(epoch_recall)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_record, epoch_acc_record, poison_identified_record, epoch_recall_record

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                # TODO: put ground truth of label besides predicted label
                ax.set_title(f'predicted: {class_names[preds[j]]}, true: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def visualize_model_test(model, num_images=6):
    """
    This function apply trained model on test data, plot them with prediction and true label
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                # TODO: put ground truth of label besides predicted label
                ax.set_title(f'predicted: {class_names[preds[j]]}, true: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def test_model(model):
    """
    This function apply trained model on test data. And return its prediction accuracuy
    """
    was_training = model.training
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            
        test_acc = running_corrects.double() / dataset_sizes['test']
        model.train(mode=was_training)
    
    return test_acc

def test_model_recall(model):
    """
    This function apply trained model on test data. And return its prediction accuracuy
    """
    was_training = model.training
    model.eval()
    recall_corrects = 0
    poison_sum = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            recall_corrects += torch.sum((preds == labels.data) * (labels.data == 1))
            poison_sum += torch.sum(labels.data == 1)
            
        test_recall = recall_corrects.double() / poison_sum
        model.train(mode=was_training)
    
    return test_recall

# Quick note about ResNet: larger image dataset with 1M+ photos and 1K+ classes

# TODO: also can try pretrained = False to retrain the resnet18 model from scratch
# should be worse than the pretrained=True as ~3000 images are still considered as small data set
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# TODO: could try another loss funciton, e.g. focal loss for classification
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# Note to use a relatively small starting learning rate, otherwise pretrained models weights are updated drastically
# i.e. we do not benefit from the existing
# TODO: could try another optimizer, e.g. Adam
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# TODO: could change lr_scheduler based on the epoch loss
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# TODO: could increase the number of epochs
model_ft_1, epoch_loss_1, epoch_acc_1, poison_identified_1, epoch_recall_1 = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,num_epochs=25)


# Save the entire model
# Specify a path
PATH = "entire_model_1.pt"

# Save
torch.save(model_ft_1, PATH)

# Load
# model = torch.load(PATH)
# model.eval()

# from the numpy module
np.savetxt("epoch_loss_18.csv", 
           epoch_loss_1,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_acc_18.csv", 
           epoch_acc_1,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_recall_18.csv", 
           epoch_recall_1,
           delimiter =", ", 
           fmt ='% s')

# resnet34
model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# TODO: could increase the number of epochs
model_ft_2, epoch_loss_2, epoch_acc_2, poison_identified_2, epoch_recall_2 = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,num_epochs=25)

# Save the entire model
# Specify a path
PATH = "entire_model_net34.pt"

# Save
torch.save(model_ft_2, PATH)

# Load
# model = torch.load(PATH)
# model.eval()

# from the numpy module
np.savetxt("epoch_loss_34.csv", 
           epoch_loss_2,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_acc_34.csv", 
           epoch_acc_2,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_recall_34.csv", 
           epoch_recall_2,
           delimiter =", ", 
           fmt ='% s')

# resnet152
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft_3, epoch_loss_3, epoch_acc_3, poison_identified_3, epoch_recall_3 = train_model(model_ft, criterion,optimizer_ft,exp_lr_scheduler,num_epochs=25)

# Save the entire model
# Specify a path
PATH = "entire_model_net152.pt"

# Save
torch.save(model_ft_3, PATH)

# Load
# model = torch.load(PATH)
# model.eval()

# from the numpy module
np.savetxt("epoch_loss_152.csv", 
           epoch_loss_3,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_acc_152.csv", 
           epoch_acc_3,
           delimiter =", ", 
           fmt ='% s')

# from the numpy module
np.savetxt("epoch_recall_152.csv", 
           epoch_recall_3,
           delimiter =", ", 
           fmt ='% s')

visualize_model_test(model_ft_1, 6)
visualize_model_test(model_ft_2, 6)
visualize_model_test(model_ft_3, 6)

test_acc_1 = test_model(model_ft_1)
test_acc_2 = test_model(model_ft_2)
test_acc_3 = test_model(model_ft_3)

# Other loss funtions
# Focal Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

model_ft = models.resnet18(pretrained=True)
#model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=5)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)  # Adam optimizer
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft, Epoch_loss, Epoch_acc,_ ,_ = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)

# Code for plotting

# Confusion matrix and test accuracy
# Reference: https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial

def confusion_matrix(model_ft, nb_classes, phase="test"):
    """
    given a pytorch model (model_ft), calculate confusion matrix (a torch tensor object)

    Arguments
    ---------
    model_ft:     pytorch model from torchvision

    nb_classes:   number of classes

    phase:        get the confusion matrix for 'test' or 'validation'
    """
    nb_classes = nb_classes
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

# plot the confusion_matrix
# Reference: https://stackoverflow.com/questions/39033880/plot-confusion-matrix-sklearn-with-multiple-labels

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2, 3]
                  the class names, for example: ['edible M', 'edible MS', 'poisonous M', 'poisonous MS']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

PATH = "entire_model_net152.pt"
# Load finetuned ResNet152 model
model152_ft = torch.load(PATH, map_location=torch.device(device))
model152_ft.eval()

# Confusion matrix for ResNet152
cm152 = confusion_matrix(model152_ft, 2)
# cols are predicted labels (left to right): edible, poisonous
# rows are true labels (left to right): edible, poisonous
print(cm152)

plot_confusion_matrix(cm152.numpy(), class_names, title='Confusion matrix for ResNet152', normalize=False)

PATH = "entire_model_net34.pt"

# Load finetuned ResNet34 model
model34_ft = torch.load(PATH, map_location=torch.device(device))
model34_ft.eval()

# Confusion matrix for ResNet34
cm34 = confusion_matrix(model34_ft, 2)
print(cm34)
plot_confusion_matrix(cm34.numpy(), class_names, title='Confusion matrix for ResNet34', normalize=False)

PATH = "entire_model_net18.pt"
# Load finetuned ResNet18 model
model18_ft = torch.load(PATH, map_location=torch.device(device))
model18_ft.eval()

# Confusion matrix for ResNet18
cm18 = confusion_matrix(model18_ft, 2)
print(cm18)
plot_confusion_matrix(cm18.numpy(), class_names, title='Confusion matrix for ResNet18', normalize=False)

# get probability instead of prediction from the model
# Reference: https://stackoverflow.com/questions/60182984/how-to-get-the-predict-probability
def predict_prob(model_ft, phase="test"):
    """
    given a pytorch model (model_ft), calculate probability for each class assignment (a torch tensor object)

    Arguments
    ---------
    model_ft:     pytorch model from torchvision


    phase:        get the confusion matrix for 'test' or 'validation'
    """
    import torch.nn.functional as nnf
    prob_predict = []
    targets = []

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            prob = nnf.softmax(outputs, dim=1)
            prob_predict.append(prob)
            targets.append(classes)

    return torch.vstack(prob_predict), torch.hstack(targets)

# Plot layers

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_load = list(model_load.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective weights to the list
for i in range(len(model_load)):
    if type(model_load[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_load[i].weight)
        conv_layers.append(model_load[i])
    elif type(model_load[i]) == nn.Sequential:
        for j in range(len(model_load[i])):
            for child in model_load[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")


data_dir = os.getcwd() + '/data2'
image_dir = data_dir+'/test/edible/ncvc (395).jpg'
from IPython.display import Image
Image(image_dir)

from PIL import Image
image = Image.open(image_dir)

transform_image = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image = transform_image(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
    
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

    
# Other machine learning models
#Logistic Regression
#K-Nearest Neighbors¶
#Random Forest
    
# obtain working directory
dataset_dir = os.getcwd() + '/data3'

# create dataset
data = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), transform[x])
                  for x in ['train', 'test']}

# obtain train and test sets
train = data['train']
test = data['test']

# obtain X and y in train set
X_train = torch.stack([img_t for img_t, _ in train], dim=0)
X_train = X_train.numpy()

y_train = np.empty(0)
y_train = np.append(y_train, [lab_t for _, lab_t in train])

# obtain X and y in test set
X_test = torch.stack([img_t for img_t, _ in test], dim=0)
X_test = X_test.numpy()

y_test = np.empty(0)
y_test = np.append(y_test, [lab_t for _, lab_t in test])

# reshape to 2D from 4D
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Logistic regression
classifier_logistic = LogisticRegression(max_iter=1000)
start_time = time.time()
classifier_logistic.fit(X_train, y_train)
print(f"Time for training model: {time.time() - start_time:.3f} seconds")

# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()
start_time = time.time()
classifier_KNN.fit(X_train, y_train)
print(f"Time for training model: {time.time() - start_time:.3f} seconds")

# Random Forest
classifier_RF = RandomForestClassifier()
classifier_RF.fit(X_train, y_train)

# use 5-fold Cross Validation
model_names = ['Logistic Regression', 'KNN', 'Random Forest']
model_list = [classifier_logistic, classifier_KNN, classifier_RF]

count = 0
for classifier in model_list:
    cv_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print(cv_score.round(3))
    print('Model accuracy of ' + model_names[count] + ' is ' + str(cv_score.mean().round(3)))
    count += 1

def cal_evaluation(classifier, conf_matrix):
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print ()
    print (classifier)
    print ("Accuracy is: " + str(accuracy.round(3)))
    print ("precision is: " + str(precision.round(3)))
    print ("recall is: " + str(recall.round(3)))

# logistic regression
lr_y_pred_prob = classifier_logistic.predict_proba(X_test)[:,1]
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_y_pred_prob)

# KNN
knn_y_pred_prob = classifier_KNN.predict_proba(X_test)[:,1]
precision_knn, recall_knn, _ = precision_recall_curve(y_test, knn_y_pred_prob)
pr_auc_knn = auc(recall_knn, precision_knn)
# ROC
fpr_knn, tpr_knn, _ = roc_curve(y_test.numpy(), knn_y_pred_prob)
# area under curve
roc_auc_knn = auc(fpr_knn, tpr_knn)

# random forest
rf_y_pred_prob = classifier_RF.predict_proba(X_test)[:,1]
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_y_pred_prob)
# area under curve
pr_auc_rf = auc(recall_rf, precision_rf)

# ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test.numpy(), rf_y_pred_prob)
# area under curve
roc_auc_rf = auc(fpr_rf, tpr_rf)

# PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_lr, precision_lr, label="LR")
plt.plot(recall_knn, precision_knn, label="KNN")
plt.plot(recall_rf, precision_rf, label="RF")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves')
plt.show()
