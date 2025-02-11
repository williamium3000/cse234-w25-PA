import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28  # maximum sequence length (e.g., the number of rows in an MNIST image)

##############################
# PART 1: Transformer Module #
##############################

def transformer(X: ad.Node, nodes: List[ad.Node], 
                model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """
    Construct the computational graph for a single transformer layer with sequence classification.
    
    The transformer is built as follows:
      1. Compute Q, K, V via three linear projections: 
           Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V.
         (Here X is of shape (batch, seq_length, input_dim) and W_Q, W_K, W_V are of shape (input_dim, model_dim).)
      2. Compute the scaled dot-product attention:
           scores = Q @ (K transposed on the last two dims)
           scaled_scores = scores / sqrt(model_dim)
           A = Softmax(scaled_scores, dim=-1)
           attn_output = A @ V.
      3. Project the attention output with W_O.
      4. Apply a two–layer feed–forward network (with a ReLU activation in between):
           F1 = ReLU( (attn_output @ W_1 + b_1) )
           F2 = F1 @ W_2 + b_2.
      5. Average the resulting sequence output along the sequence (dim=1) to yield (batch, num_classes).
    
    The parameter nodes are created as ad.Variable nodes and appended to the list “nodes”.
    """
    # Create parameter nodes for all linear projections and feed-forward layers.
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1 = ad.Variable("W_1")
    b_1 = ad.Variable("b_1")
    W_2 = ad.Variable("W_2")
    b_2 = ad.Variable("b_2")
    
    # Append parameters so they can be later mapped to concrete tensor values.
    nodes.extend([W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2])
    
    # --- Self-Attention Block ---
    # Compute Q, K, V (each using a linear projection)
    Q = ad.matmul(X, W_Q)  # shape: (batch, seq_length, model_dim)
    K = ad.matmul(X, W_K)
    V = ad.matmul(X, W_V)
    
    # Transpose K along the last two dimensions so that we can perform matmul.
    K_T = ad.transpose(K, 1, 2)  # shape: (batch, model_dim, seq_length)
    
    # Compute attention scores and scale them.
    scores = ad.matmul(Q, K_T)            # shape: (batch, seq_length, seq_length)
    scale = model_dim ** 0.5              # scaling factor: sqrt(model_dim)
    scaled_scores = ad.div_by_const(scores, scale)
    
    # Apply softmax along the last dimension.
    A = ad.softmax(scaled_scores, dim=-1)
    
    # Multiply the attention weights with V.
    attn_output = ad.matmul(A, V)         # shape: (batch, seq_length, model_dim)
    
    # Project the attention output.
    O = ad.matmul(attn_output, W_O)        # shape: (batch, seq_length, model_dim)
    
    # --- Feed-Forward Network ---
    # First linear layer: (attn_output projected) @ W_1 + b_1.
    F1 = ad.add(ad.matmul(O, W_1), b_1)
    F1_relu = ad.relu(F1)
    # Second linear layer: F1_relu @ W_2 + b_2.
    F2 = ad.add(ad.matmul(F1_relu, W_2), b_2)  # shape: (batch, seq_length, num_classes)
    
    # Average over the sequence dimension (dim=1) to get final logits.
    output = ad.mean(F2, dim=1, keepdim=False)  # shape: (batch, num_classes)
    return output

#################################
# PART 2: Loss and SGD Training #
#################################

def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """
    Construct the computational graph of average softmax loss over a batch.
    
    We compute the loss as:
    
         loss = - (1/batch_size) * sum( y_one_hot * log(softmax(Z)) )
    """
    softmax_Z = ad.softmax(Z, dim=-1)
    log_softmax_Z = ad.log(softmax_Z)
    # Sum along the class dimension.
    cross_entropy_loss = ad.sum_op(y_one_hot * log_softmax_Z, dim=1)
    loss = ad.sum_op(cross_entropy_loss, dim=0)
    avg_loss = ad.div_by_const(loss, batch_size)
    avg_loss = ad.mul_by_const(avg_loss, -1)
    return avg_loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """
    Run one epoch of SGD.
    
    For each mini–batch, we run the forward and backward passes (using f_run_model),
    update every model parameter by subtracting the learning rate times its gradient,
    and accumulate the training loss.
    """
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        # Get mini-batch indices.
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        # Select mini-batch data.
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Forward and backward pass.
        # f_run_model should return [logits, loss, grad_W_Q, grad_W_K, ..., grad_b2]
        logits, loss_val, *grads = f_run_model(model_weights, X_batch, y_batch)

        # Update each model parameter.
        for j in range(len(model_weights)):
            model_weights[j] = model_weights[j] - lr * grads[j]
        
        # Multiply by number of examples in batch to get total loss.
        total_loss += loss_val.item() * (end_idx - start_idx)

    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)
    return model_weights, average_loss


###############################
# PART 3: Complete Training   #
###############################

def train_model():
    """
    Train a single-layer transformer (ViT–style) on MNIST.
    
    The forward graph consists of:
      - An input placeholder X (of shape (batch, seq_length, input_dim))
      - A ground truth placeholder y
      - The transformer layer (which creates its own parameter nodes)
      - A softmax loss over the averaged sequence output.
      
    The backward graph is built using the gradients() function.
    """
    # Hyperparameters.
    input_dim = 28       # each row of the image has 28 pixels
    seq_length = max_len # number of rows (i.e., sequence length) = 28
    num_classes = 10
    model_dim = 128      # hidden dimension
    eps = 1e-5 
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # --- Build the forward graph ---
    # Create input placeholders.
    X_node = ad.Variable("X")
    y_groundtruth = ad.Variable("y")
    
    # Create an empty list for parameter nodes.
    nodes = []
    # Build the transformer forward pass.
    y_predict = transformer(X_node, nodes, model_dim, seq_length, eps, batch_size, num_classes)
    # Compute softmax loss.
    loss = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # --- Build the backward graph ---
    # Compute gradients for every parameter in the transformer.
    grads = ad.gradients(loss, nodes)
    
    # Create evaluators for training (forward + backward) and testing (forward only).
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])
    
    # --- Load the MNIST dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # One-hot encode the training labels.
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    
    # --- Initialize Model Weights ---
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    # The ordering must match the order in which the transformer() function appended the parameters.
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val, dtype=torch.double),
        torch.tensor(W_K_val, dtype=torch.double),
        torch.tensor(W_V_val, dtype=torch.double),
        torch.tensor(W_O_val, dtype=torch.double),
        torch.tensor(W_1_val, dtype=torch.double),
        torch.tensor(b_1_val, dtype=torch.double),
        torch.tensor(W_2_val, dtype=torch.double),
        torch.tensor(b_2_val, dtype=torch.double)
    ]
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    # --- Define f_run_model and f_eval_model ---
    def f_run_model(model_weights, X_batch, y_batch):
        # Create the mapping from our AD variables to the actual tensor values.
        mapping = { X_node: X_batch, y_groundtruth: y_batch }
        for i, param in enumerate(nodes):
            mapping[param] = model_weights[i]
        # evaluator.run returns [y_predict, loss, grad_W_Q, grad_W_K, ...]
        return evaluator.run(mapping)
    
    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            mapping = { X_node: X_batch }
            for i, param in enumerate(nodes):
                mapping[param] = model_weights[i]
            logits = test_evaluator.run(mapping)
            all_logits.append(logits[0])
        # Concatenate all logits (predictions) and choose the max along the class dimension.
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions
    
    # --- Training Loop ---
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(f_run_model, X_train, y_train, model_weights, batch_size, lr)
        predict_label = f_eval_model(X_test, model_weights)
        test_accuracy = np.mean(predict_label == y_test)
        print(f"Epoch {epoch}: test accuracy = {test_accuracy}, loss = {loss_val}")
    
    # Final evaluation.
    predict_label = f_eval_model(X_test, model_weights)
    final_accuracy = np.mean(predict_label == y_test)
    return final_accuracy

if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
