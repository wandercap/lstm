import os
import time
import data_loader
import itertools
from torchtext import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from lstm import LSTMClassifier

TEXT, LABEL, label_size, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = data_loader.load_dataset()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)   

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()

    optim.zero_grad()
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.summary[0]
        target = batch.rating
        target = torch.autograd.Variable(target).long()

        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):
            continue

        optim.zero_grad()

        prediction = model(text)

        loss = loss_fn(prediction, target)

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)

        loss.backward()
        # clip_gradient(model, 1e-1)

        optim.step()
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, test):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    model.eval()

    num_batches = len(val_iter)
    num_elements = len(val_iter.dataset)

    predictions = torch.zeros(num_elements, 4)
    targets = torch.zeros(num_elements)
    
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            start = idx*batch_size
            end = start + batch_size
            text = batch.summary[0]
            target = batch.rating
            target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            if (text.size()[0] is not 32):
                continue

            prediction = model(text)

            if idx == num_batches - 1:
                end = num_elements

            predictions[start:end] = prediction  
            targets[start:end] = target

            loss = loss_fn(prediction, target)

            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    if (test):

        _, preds = torch.max(predictions, 1)
        np_target = targets.cpu()
        np_target = np_target.numpy()
        np_preds = preds.cpu()
        np_preds = np_preds.numpy()

        accuracy = accuracy_score(np_target, np_preds)
        print('Acurácia: ' + str(accuracy))

        recall = recall_score(np_target, np_preds, average='weighted')
        print('Recall: ' + str(recall))

        precision = precision_score(np_target, np_preds, average='weighted')
        print('Precisão: ' + str(precision))

        f1 = f1_score(np_target, np_preds, average='weighted')
        print('F1 Score: ' + str(f1))

        cnf_matrix = confusion_matrix(np_target, np_preds)

        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=LABEL.vocab.itos, normalize=True,
                              title='Confusion matrix')

        plt.show()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


input_size = vocab_size
output_size = label_size
hidden_size = 128
embedding_length = 300
num_layers = 2
bidirectional = True
dropout = 0.5
num_epochs = 10
learning_rate = 0.001
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(batch_size, input_size, embedding_length, hidden_size, output_size, num_layers, bidirectional, dropout)

pretrained_embeddings = TEXT.vocab.vectors

model.word_embeddings.weight.data.copy_(pretrained_embeddings)

optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

model = model.to(device)
loss_fn = loss_fn.to(device)

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter, False)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

test_loss, test_acc = eval_model(model, test_iter, True)
print(f'Test Loss: {test_loss:3f}, Test Acc: {test_acc:.2f}%')

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(test_iter.dataset.examples, preds)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=LABEL.vocab,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=LABEL.vocab, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()

torch.save(model.state_dict(), 'model/model.pth')