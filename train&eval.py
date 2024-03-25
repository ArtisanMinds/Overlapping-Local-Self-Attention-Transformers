import torch
import torch.jit as jit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# define the training function
def train_model(model, train_loader, validation_loader, criterion, optimizer, model_name,
                epochs, device, scheduler_plateau, patience):
    # set the model to training mode
    model.train()
    # initialize the best accuracy and validation loss for early stopping
    best_accuracy = float('-inf')
    # current_accuracy = float('-inf')
    epochs_no_improve = 0
    # loop over the number of epochs
    for epoch in range(epochs):
        # iterate over each batch in the training dataset
        for features, labels, mask in train_loader:
            # move data to the GPU
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            # zero the gradients before running the backward pass
            optimizer.zero_grad()
            # forward pass through the model
            outputs = model(features, mask)
            # reshape labels to match the output shape
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            # apply mask to filter out padded values for loss calculation
            valid_mask = labels != -1
            loss = criterion(outputs[valid_mask], labels[valid_mask])
            # backward pass
            loss.backward()
            # update model parameters
            optimizer.step()
        # evaluate the model on the validation dataset
        val_loss, val_accuracy, val_precision, val_recall, val_f1 \
            = evaluate_model(model, validation_loader, criterion, device)
        # print training and validation metrics
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, '
              f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, '
              f'Validation F1 Score: {val_f1}, Validation Precision: {val_precision}, Validation Recall: {val_recall}')

        # check for early stopping criteria based on validation loss and accuracy
        if round(val_accuracy, 5) > round(best_accuracy, 5):
            best_accuracy = val_accuracy
            epochs_no_improve = 0
            # save the best model by jit
            jit.save(model, model_name)
        else:
            epochs_no_improve += 1
            # stop training if there is no improvement for 'patience' number of epochs
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break
        # update the learning rate based on the validation accuracy
        scheduler_plateau.step(val_accuracy)
        # sure that the learning rate is not too small
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-6:
                param_group['lr'] = 1e-6


# define the evaluation function for the best model
def evaluate_model(model, loader, criterion, device):
    # set the model to evaluation mode
    model.eval()
    # initialize metrics to 0
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    valid_labels_list = []
    with torch.no_grad():  # disable gradient calculations for evaluation
        # iterate over each batch in the validation dataset
        for features, labels, mask in loader:
            # move data to GPU
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            # forward pass through the model
            outputs = model(features, mask)
            # reshape for calculating loss and accuracy
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            # apply mask to filter out padded values for loss calculation
            valid_mask = labels != -1
            valid_outputs = outputs[valid_mask]
            valid_labels = labels[valid_mask]
            # calculate loss
            loss = criterion(valid_outputs, valid_labels)
            total_loss += loss.item() * valid_labels.size(0)
            # calculate accuracy
            _, predicted = torch.max(valid_outputs, 1)
            total_correct += (predicted == valid_labels).sum().item()
            total_samples += valid_labels.size(0)
            # collect all predictions and labels for computing precision, recall, and F1 score
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            # use softmax to get the probability distribution
            probabilities = torch.softmax(valid_outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            valid_labels_list.extend(valid_labels.cpu().numpy())
    # calculate average loss, accuracy, precision, recall, F1 score, and confusion matrix
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    confusion = confusion_matrix(all_labels, all_predictions)
    # return all the computed metrics
    return (avg_loss, accuracy, precision, recall, f1, confusion, valid_labels_list, all_probabilities)
