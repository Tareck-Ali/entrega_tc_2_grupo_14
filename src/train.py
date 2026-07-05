from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from classes import EmbeddingRecommender, InteractionDataset
from preprocessing import preprocess
from torch.optim import Adam
from torch import nn
import torch
from pathlib import Path
import mlflow
import joblib

def prepare_data():
    """
    Prepares training and validation data loaders for the recommendation system.

    This function:
    - Loads and preprocesses the raw interaction data via `preprocess()`
    - Encodes users and items into integer indices
    - Splits the dataset into training and validation sets
    - Wraps the splits into `InteractionDataset` objects
    - Creates PyTorch DataLoaders for batching

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): Full preprocessed dataset with encoded user/item indices
            - user_encoder: Encoder used to transform user IDs into indices
            - item_encoder: Encoder used to transform item IDs into indices
            - train_loader (DataLoader): Training data loader
            - val_loader (DataLoader): Validation data loader
    """
    df, user_encoder, item_encoder = preprocess()

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    train_dataset = InteractionDataset(train_df)
    val_dataset = InteractionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    return df, user_encoder, item_encoder, train_loader, val_loader

def build_model(df):
    """
    Builds the embedding-based recommender model along with optimizer and loss function.

    The model learns user and item embeddings to predict interaction scores.

    Args:
        df (pd.DataFrame): Preprocessed dataset containing at least 'user_idx' and 'item_idx' columns.

    Returns:
        tuple: A tuple containing:
            - model (EmbeddingRecommender): Initialized recommendation model
            - optimizer (torch.optim.Optimizer): Adam optimizer for training
            - criterion (torch.nn.Module): Loss function (MSELoss)
    """
    model = EmbeddingRecommender(
        num_users=df["user_idx"].nunique(),
        num_items=df["item_idx"].nunique(),
        embedding_dim=64,
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    return model, optimizer, criterion

def train_epoch(model, loader, optimizer, criterion):
    """
    Trains the model for one epoch over the provided dataset.

    Performs forward pass, loss computation, backpropagation, and optimizer updates.

    Args:
        model (torch.nn.Module): Recommendation model
        loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates
        criterion (torch.nn.Module): Loss function

    Returns:
        float: Average training loss across all batches in the epoch
    """
    model.train()
    total_loss = 0.0

    for users, items, ratings in loader:
        preds = model(users, items)
        loss = criterion(preds, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion):
    """
    Evaluates the model on the validation dataset.

    Runs inference without gradient computation and computes average loss.

    Args:
        model (torch.nn.Module): Trained recommendation model
        loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function

    Returns:
        float: Average validation loss across all batches
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for users, items, ratings in loader:
            preds = model(users, items)
            loss = criterion(preds, ratings)
            total_loss += loss.item()

    return total_loss / len(loader)

def setup_mlflow():
    """
    Configures and starts an MLflow experiment run.

    This function:
    - Sets the experiment name to "recommendation-system"
    - Starts and returns an MLflow run context

    Returns:
        mlflow.ActiveRun: Active MLflow run context manager
    """
    mlflow.set_experiment("recommendation-system")
    return mlflow.start_run()

def save_artifacts(user_encoder, item_encoder, model):
    """
    Saves model artifacts and logs them to MLflow.

    This function:
    - Creates a local artifacts directory if it does not exist
    - Serializes user and item encoders using joblib
    - Logs encoders as artifacts in MLflow
    - Logs the trained PyTorch model to MLflow

    Args:
        user_encoder: Encoder used for user ID transformation
        item_encoder: Encoder used for item ID transformation
        model (torch.nn.Module): Trained recommendation model
    """
    Path("artifacts").mkdir(exist_ok=True)

    joblib.dump(user_encoder, "artifacts/user_encoder.pkl")
    joblib.dump(item_encoder, "artifacts/item_encoder.pkl")

    mlflow.log_artifact("artifacts/user_encoder.pkl")
    mlflow.log_artifact("artifacts/item_encoder.pkl")

    mlflow.pytorch.log_model(model, artifact_path="model")

def train() -> None:
    """
    Runs the full training pipeline for the recommendation system.

    This function:
    - Prepares data loaders and encoders
    - Builds the model, optimizer, and loss function
    - Initializes MLflow tracking
    - Trains the model for a fixed number of epochs (10)
    - Logs training/validation metrics to MLflow
    - Saves trained model and encoders as artifacts

    Returns:
        None
    """
    df, user_enc, item_enc, train_loader, val_loader = prepare_data()

    model, optimizer, criterion = build_model(df)

    with setup_mlflow():

        mlflow.log_params({
            "embedding_dim": 64,
            "batch_size": 256,
            "epochs": 10,
            "learning_rate": 1e-3,
        })

        for epoch in range(10):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)

            print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} val={val_loss:.4f}")

        save_artifacts(user_enc, item_enc, model)
