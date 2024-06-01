import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_size, input_size * 2//3)

        # Optionally add more layers
        self.fc2 = nn.Linear(input_size * 2//3, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        # Apply the first fully connected layer and a ReLU activation
        x = F.relu(self.fc1(x))

        # Apply the output layer
        return self.fc2(x)


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.conv2d1 = nn.Conv2d(input_channels, 6, 3, padding=1)
        self.conv2d2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = x.flatten(-3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                # Calculate the dot product between queries and keys, and scale it.
                scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

                # Apply softmax to get the attention weights.
                attention = self.softmax(scaled_dot_product)

                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out
      
class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) # patch size in terms of height and width

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, c, h, w = x.shape

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out
    
    @staticmethod
    def patchify(images, n_patches):
        n, c, h, w = images.shape

        assert h == w # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):

                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()

        return patches
    
    @staticmethod
    def get_positional_embeddings(sequence_length, d_model):
        # Initialize the result tensor with the correct shape
        result = torch.zeros(sequence_length, d_model)

        # Precompute the denominators for efficiency
        denominator = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)

        # Loop through each position in the sequence
        for i in range(sequence_length):
            # Calculate the values for even indices
            result[i, 0::2] = torch.sin(i / denominator)
        
            # Calculate the values for odd indices (check if there is an odd index available)
            if d_model % 2 == 1:
                # If d_model is odd, the last index will be out of range for cos, so handle separately
                result[i, 1:-1:2] = torch.cos(i / denominator[:-1])  # Avoid the last index
                result[i, -1] = torch.cos(i / denominator[-1])  # Handle the last index separately
            else:
                # If d_model is even, we can safely use the slicing
                result[i, 1::2] = torch.cos(i / denominator)

        return result


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()

        # Determine the optimizer based on the type of model
        if isinstance(model, (MLP, CNN)):
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        train_loss = 0.0
        total_accuracy = 0.0  # To average accuracy over iterations if needed
        for it, batch in enumerate(dataloader):

            # Load a batch, break it down in images and targets.
            x, y = batch

            # Run forward pass.
            logits = self.model(x)

            # Compute loss (using 'criterion').
            loss = self.criterion(logits, y)

            # Run backward pass.
            loss.backward()

            train_loss += loss.detach().cpu().item() / len(dataloader)

            # Update the weights using 'optimizer'.
            self.optimizer.step()

            # Zero-out the accumulated gradients.
            self.optimizer.zero_grad()

            # Compute accuracy
            acc = self.accuracy(logits, y)
            total_accuracy += acc

            # Optionally, print training progress
            #print(f'\rEpoch {ep+1}/{self.epochs}, Iteration {it+1}/{len(dataloader)}: Loss = {loss.item():.4f}, Accuracy = {acc:.2f}%', end='')

        # Print average loss and accuracy for the epoch if needed
        #average_loss = train_loss / len(dataloader)
        #average_accuracy = total_accuracy / len(dataloader)
        #print(f'\nEpoch {ep+1} completed. Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.2f}%')
            

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.extend(predicted.tolist())
        pred_labels = torch.tensor(pred_labels)  # Convert list to a tensor
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        start_time = time.time()  # Start timing

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        end_time = time.time()  # End timing
        print(f"Execution Time : {end_time - start_time:.4f} seconds")  # Print the execution time
        
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
    

    def accuracy(self, logits, y):
        """
        Calculate the accuracy of predictions.

        Args:
            logits (torch.Tensor): The logits output from the model.
            y (torch.Tensor): The ground truth labels.

        Returns:
            float: The accuracy percentage.
        """
        predicted = torch.argmax(logits, dim=1)
        correct = (predicted == y).type(torch.float)
        accuracy = correct.mean().item() * 100
        return accuracy