import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' # set max_split_size_mb to reduce reserved unused memory
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
from sklearn.decomposition import PCA # For PCA analysis
# Class imports from seperate files #
from dataclass_semantic import Dataclass
from unet_semanticv2 import UNet
from domain_adaptation_modelv2 import DomainAdaptationModel
from domain_classifierv2 import DomainClassifier
import lovasz_losses as L

def binary_accuracy(preds, labels):
    """
    Calculates accuracy for binary predictions.
    
    Args: 
    preds: Predictions from the model, as a tensor.
    labels: Ground truth binary labels, as a tensor.
    
    Returns:
    Accuracy as a float.
    """
    # Apply threshold to binary predictions
    thresholded_preds = preds >= 0.5  # Convert probabilities to binary predictions

    # Compare with true labels
    correct = (thresholded_preds == labels).float()  # Convert boolean to float for calculation
    accuracy = correct.sum() / len(correct)  # Calculate accuracy
    return accuracy.item()  # Return as a Python float

def get_current_hyperparams():
    hyperparams_str = f"Learning Rate: {learning_rate}\nWeight_decay: {weight_decay}\nBatch size: {batch_size}\nDropout: {unet_model.dropout}\n"
    return hyperparams_str

# Initialize a list to store metrics strings
metrics_list = []
def log_training(epoch, train_seg_loss, train_domain_loss, train_f1, train_iou, val_seg_loss,
                 val_domain_loss, val_f1, val_iou, train_domain_acc,
                 val_domain_acc, log_file="Metrics.txt"):
    """
    Logs training and validation metrics for each epoch, including hyperparameters.

    Parameters:
    - epoch (int): Current epoch number.
    - train_seg_loss (float): Average segmentation loss on the training set.
    - train_domain_loss (float): Average domain classification loss on the training set.
    - train_f1 (float): Average F1 score for segmentation on the training set.
    - train_iou (float): Average Intersection over Union (IoU) for segmentation on the training set.
    - val_seg_loss (float): Average segmentation loss on the validation set.
    - val_domain_loss (float): Average domain classification loss on the validation set.
    - val_f1 (float): Average F1 score for segmentation on the validation set.
    - val_iou (float): Average IoU for segmentation on the validation set.
    - train_domain_acc (float): Average domain classification accuracy on the training set.
    - val_domain_acc (float): Average domain classification accuracy on the validation set.
    - log_file (str): Path to the log file.
    """
    metrics_str = f"Epoch: {epoch}, " \
                  f"Training Segmentation Loss: {train_seg_loss:.4f}, Training Domain Loss: {train_domain_loss:.4f}, " \
                  f"Training Segmentation F1: {train_f1:.4f}, Training Segmentation IoU: {train_iou:.4f}, " \
                  f"Training Domain Accuracy: {train_domain_acc:.4f}, " \
                  f"Validation Segmentation Loss: {val_seg_loss:.4f}, Validation Domain Loss: {val_domain_loss:.4f}, " \
                  f"Validation Segmentation F1: {val_f1:.4f}, Validation Segmentation IoU: {val_iou:.4f}, " \
                  f"Validation Domain Accuracy: {val_domain_acc:.4f}\n"
    
    if epoch == 1:
        hyperparams = get_current_hyperparams()
        with open(log_file, "a") as file:
            file.write("Hyperparameters:\n" + hyperparams)  # Adding a header for hyperparameters

    # Append metrics to the list
    metrics_list.append(metrics_str)

    # Write the accumulated metrics to the file at specified intervals and at the end of training
    if epoch % 10 == 0 or epoch == num_epochs:
        with open(log_file, "a") as file:
            for metric in metrics_list:
                file.write(metric)
            metrics_list.clear()

# Enable TF32 for improved performance on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set random seed for reproducibility
torch.manual_seed(42)

# Step 1: Setup Dataset
dataset = Dataclass(
    simulated_image_dir='/zhome/10/0/181468/semantic/NEW_IMAGES',
    real_image_dir = '/zhome/10/0/181468/semantic/domain_adaptation/all_train_processed',
    simulated_label_dir='/zhome/10/0/181468/semantic/NEW_PROCESSED_SEMANTICS')

# Define the sizes for training, validation, and test sets
train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.2 * len(dataset))    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for test

# Split dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each set
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Step 2: Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet(n_class=7).to(device) # Total of 7 classes = {Red, Pink, Yellow, Blue, Cyan, Green, Black}
domain_classifier = DomainClassifier(input_features=2048*21*12).to(device) # Number of filters * Width * Height
model = DomainAdaptationModel(unet_model, domain_classifier).to(device)

# Step 3: Loss Function & Optimizer
segmentation_criterion = L.lovasz_softmax 
domain_criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 0.00017587407735889307
weight_decay = 0.052909730367861826
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Function for PCA analysis
def save_pca_data(features, labels, epoch, output_dir='pca_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features_flat = features.reshape(features.shape[0], -1)
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features_flat)

    # Save the PCA features and labels as .npy files
    np.save(os.path.join(output_dir, f'pca_features_epoch_{epoch}.npy'), pca_features)
    np.save(os.path.join(output_dir, f'labels_epoch_{epoch}.npy'), labels)

# Extract features for visualizing features in the PCA 
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, _, domain_labels in dataloader:
            images = images.to(device)
            # Get features before gradient reversal
            _, feature_output = model.unet_model(images)
            features.append(feature_output.cpu().numpy())
            labels.append(domain_labels.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


# Initialize lists to store metrics
train_losses = []
val_losses = []
train_f1_scores = []
train_ious = []
val_f1_scores = []
val_ious = []

# Step 4: Training Loop
num_epochs = 35 
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_seg_loss_accum = 0.0
    total_domain_loss_accum = 0.0
    train_f1_accum = 0
    train_iou_accum = 0
    train_domain_acc_accum = 0  # Initialize accumulator for domain accuracy
    
    for batch_idx, (images, segmentation_labels, domain_labels) in enumerate(train_loader):
        images = images.to(device)
        segmentation_labels = segmentation_labels.to(device)
        domain_labels = domain_labels.to(device).float()
        optimizer.zero_grad()  # zero the parameter gradients

        # Forward pass through the model    
        segmentation_pred, domain_pred = model(images)
        
        # Calculate domain classification accuracy
        domain_accuracy = binary_accuracy(domain_pred.squeeze(1), domain_labels)
        train_domain_acc_accum += domain_accuracy
        # Segmentation and domain classification loss
        seg_loss = segmentation_criterion(segmentation_pred, segmentation_labels)
        domain_loss = domain_criterion(domain_pred.squeeze(1), domain_labels)
        # Combine the losses
        total_loss = seg_loss + domain_loss

        # Backward pass
        total_loss.backward()
        # Gradient clippping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0, norm_type=2.0) # Experiment with max_norm, norm_type is the default L2.
        optimizer.step()

        # Accumulate losses for logging
        total_seg_loss_accum += seg_loss.item()
        total_domain_loss_accum += domain_loss.item()

        # Convert model outputs to discrete predictions
        predictions = torch.argmax(segmentation_pred, dim=1).detach().cpu().numpy()
        segmentation_labels_numpy = segmentation_labels.squeeze(1).long().detach().cpu().numpy()  # Convert labels to numpy
        # Flatten the arrays to 1D for metric calculations
        labels_flat = segmentation_labels_numpy.flatten()
        predictions_flat = predictions.flatten()
        # Calculate and accumulate F1-score and IoU
        f1 = f1_score(labels_flat, predictions_flat, average='macro')
        iou = jaccard_score(labels_flat, predictions_flat, average='macro')
        train_f1_accum += f1
        train_iou_accum += iou
        
        # Batch losses for printing
        average_seg_loss_so_far = total_seg_loss_accum / (batch_idx + 1)
        average_domain_loss_so_far = total_domain_loss_accum / (batch_idx + 1)

        if batch_idx % 10 == 0:  # print every 10 batches
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                f"Total Loss: {total_loss.item()}, "
                f"Avg Seg Loss So Far: {average_seg_loss_so_far}, "
                f"Avg Domain Loss So Far: {average_domain_loss_so_far}")
            
    # PCA call        
    features, labels = extract_features(model, train_loader, device) 
    save_pca_data(features, labels, epoch)
    # Averaging metrics over the epoch
    average_train_seg_loss = total_seg_loss_accum / len(train_loader)
    average_train_domain_loss = total_domain_loss_accum / len(train_loader)
    average_train_f1 = train_f1_accum / len(train_loader)
    average_train_iou = train_iou_accum / len(train_loader)
    average_train_domain_acc = train_domain_acc_accum / len(train_loader)
    # Logging epoch-level metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Seg Loss: {average_train_seg_loss}, Avg Domain Loss: {average_train_domain_loss}, Avg F1: {average_train_f1}, Avg IoU: {average_train_iou}, Avg Domain Acc: {average_train_domain_acc}")

    # VALIDATION PHASE
    model.eval()  # Set model to evaluation mode
    val_f1_accum = 0
    val_iou_accum = 0
    val_domain_acc_accum = 0
    val_seg_loss_accum = 0  # Accumulator for segmentation loss during validation
    val_domain_loss_accum = 0  # Accumulator for domain loss during validation
    valid_seg_batches = 0  # Counter for batches with valid segmentation labels

    with torch.no_grad():
        for images, segmentation_labels, domain_labels in val_loader:
            images = images.to(device)
            segmentation_labels = segmentation_labels.to(device)
            domain_labels = domain_labels.to(device).float()

            # Forward pass
            segmentation_pred, domain_pred = model(images)

            # Calculate domain classification accuracy for validation
            domain_accuracy = binary_accuracy(domain_pred.squeeze(1), domain_labels)
            
            val_domain_acc_accum += domain_accuracy
            # Compute domain classification loss
            val_domain_loss = domain_criterion(domain_pred.squeeze(1), domain_labels) 
            val_domain_loss_accum += val_domain_loss.item()

            # Check if the current batch has valid segmentation labels
            if domain_labels[0] == 0:
                # Compute validation losses
                val_seg_loss = segmentation_criterion(segmentation_pred, segmentation_labels)
                val_seg_loss_accum += val_seg_loss.item()
                
                # Convert model outputs to discrete predictions for metric calculations
                predictions = torch.argmax(segmentation_pred, dim=1).detach().cpu().numpy()
                true_labels = segmentation_labels.squeeze(1).detach().cpu().numpy() # Convert to numpy

                # Flatten arrays for F1-score and IoU calculations
                predictions_flat = predictions.flatten()
                true_labels_flat = true_labels.flatten()
                # Calculate and accumulate F1-score and IoU
                f1 = f1_score(true_labels_flat, predictions_flat, average='macro')
                iou = jaccard_score(true_labels_flat, predictions_flat, average='macro')
                val_f1_accum += f1
                val_iou_accum += iou

                valid_seg_batches += 1  # Increment counter
        
        # Calculate average validation loss, F1-score, mIoU and domain accuracy for the epoch
        average_val_seg_loss = val_seg_loss_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        average_val_domain_loss = val_domain_loss_accum / len(val_loader)
        average_val_f1 = val_f1_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        average_val_iou = val_iou_accum / valid_seg_batches if valid_seg_batches > 0 else 0
        average_val_domain_acc = val_domain_acc_accum / len(val_loader)
        # Logging validation metrics
        print(f'Validation\nEpoch {epoch+1}: Seg Loss: {average_val_seg_loss}, Domain Loss: {average_val_domain_loss}, F1-score: {average_val_f1}, mIoU: {average_val_iou}, Domain Accuracy: {average_val_domain_acc}') # ALL ARE AVERAGES

        log_training(epoch + 1, average_train_seg_loss, average_train_domain_loss, average_train_f1, average_train_iou, average_val_seg_loss, average_val_domain_loss, average_val_f1, average_val_iou, average_train_domain_acc, average_val_domain_acc)
# Save the model after training
torch.save(model.state_dict(), 'Model.pth')
print("Finished Training and saved the model.")