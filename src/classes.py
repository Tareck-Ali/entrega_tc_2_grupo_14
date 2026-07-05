from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


EVENT_WEIGHTS = { # settings?
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interaction records.

    Each sample consists of a user index, an item index, and the
    corresponding interaction rating used as the training target.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initializes the interaction dataset.

        Args:
            dataframe: DataFrame containing the encoded user indices,
                item indices, and interaction ratings. The DataFrame
                must contain the columns ``user_idx``, ``item_idx``,
                and ``rating``.
        """
        self.users = torch.tensor(
            dataframe["user_idx"].values,
            dtype=torch.long,
        )

        self.items = torch.tensor(
            dataframe["item_idx"].values,
            dtype=torch.long,
        )

        self.targets = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Returns the number of interaction samples.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.users)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Retrieves a single interaction sample.

        Args:
            index: Position of the sample to retrieve.

        Returns:
            A tuple containing the user index, item index, and target
            rating.
        """

        return (
            self.users[index],
            self.items[index],
            self.targets[index],
        )


class EmbeddingRecommender(nn.Module):
    """Matrix factorization recommender using embedding layers.

    The model learns latent representations for users and items. The
    predicted interaction score is computed as the dot product between
    the corresponding user and item embeddings.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ) -> None:
        """Initializes the recommendation model.

        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            embedding_dim: Size of the latent embedding vectors.
        """

        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

    def forward(
        self,
        users: Tensor,
        items: Tensor,
    ) -> Tensor:
        """Computes predicted interaction scores.

        Args:
            users: Tensor containing user indices.
            items: Tensor containing item indices.

        Returns:
            Tensor containing the predicted interaction score for each
            user-item pair.
        """

        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)

        return (user_vectors * item_vectors).sum(dim=1)


def preprocess(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Preprocesses the interaction dataset.

    Event types are converted into numeric ratings, while user and item
    identifiers are label encoded for use in embedding layers.

    Args:
        dataframe: Raw interaction DataFrame containing at least the
            columns ``visitorid``, ``itemid``, and ``event``.

    Returns:
        A tuple containing:
            - The processed DataFrame.
            - The fitted user label encoder.
            - The fitted item label encoder.
    """

    dataframe = dataframe.copy()

    dataframe["rating"] = dataframe["event"].map(EVENT_WEIGHTS)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    dataframe["user_idx"] = user_encoder.fit_transform(
        dataframe["visitorid"]
    )

    dataframe["item_idx"] = item_encoder.fit_transform(
        dataframe["itemid"]
    )

    return dataframe, user_encoder, item_encoder


def train() -> None:

    """Trains the recommendation model.

    The training pipeline performs the following steps:

    - Loads the interaction dataset.
    - Preprocesses user and item identifiers.
    - Splits the data into training and validation sets.
    - Creates PyTorch datasets and data loaders.
    - Trains the embedding-based recommendation model.
    - Logs parameters, metrics, and artifacts with MLflow.
    - Saves the trained model and fitted label encoders.
    """

    ####################################################################
    # Load data
    ####################################################################

    df = pd.read_csv("events.csv")

    ####################################################################
    # Preprocess
    ####################################################################

    df, user_encoder, item_encoder = preprocess(df)

    ####################################################################
    # Split
    ####################################################################

    train_df, validation_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    ####################################################################
    # Dataset
    ####################################################################

    train_dataset = InteractionDataset(train_df)
    validation_dataset = InteractionDataset(validation_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=256,
    )

    ####################################################################
    # Model
    ####################################################################

    model = EmbeddingRecommender(
        num_users=df["user_idx"].nunique(),
        num_items=df["item_idx"].nunique(),
        embedding_dim=64,
    )

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.MSELoss()

    ####################################################################
    # MLflow
    ####################################################################

    mlflow.set_experiment("recommendation-system")

    with mlflow.start_run():

        mlflow.log_param("embedding_dim", 64)
        mlflow.log_param("batch_size", 256)
        mlflow.log_param("epochs", 10)
        mlflow.log_param("learning_rate", 1e-3)

        ################################################################
        # Training
        ################################################################

        epochs = 10

        for epoch in range(epochs):

            model.train()

            train_loss = 0.0

            for users, items, ratings in train_loader:

                predictions = model(users, items)

                loss = criterion(
                    predictions,
                    ratings,
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            ############################################################
            # Validation
            ############################################################

            model.eval()

            validation_loss = 0.0

            with torch.no_grad():

                for users, items, ratings in validation_loader:

                    predictions = model(users, items)

                    loss = criterion(
                        predictions,
                        ratings,
                    )

                    validation_loss += loss.item()

            validation_loss /= len(validation_loader)

            ############################################################
            # MLflow Metrics
            ############################################################

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "validation_loss",
                validation_loss,
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1:02d} | "
                f"train={train_loss:.4f} "
                f"validation={validation_loss:.4f}"
            )

        ################################################################
        # Save artifacts
        ################################################################

        Path("artifacts").mkdir(exist_ok=True)

        joblib.dump(
            user_encoder,
            "artifacts/user_encoder.pkl",
        )

        joblib.dump(
            item_encoder,
            "artifacts/item_encoder.pkl",
        )

        mlflow.log_artifact(
            "artifacts/user_encoder.pkl"
        )

        mlflow.log_artifact(
            "artifacts/item_encoder.pkl"
        )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
        )


def recommend(
    model: EmbeddingRecommender,
    visitor_id: int,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    top_k: int = 10,
) -> list[int]:
    """Generates item recommendations for a user.

    Recommendations are produced by computing the similarity between the
    user's embedding and all item embeddings, then selecting the highest
    scoring items.

    Args:
        model: Trained recommendation model.
        visitor_id: Original user identifier.
        user_encoder: Label encoder used for user identifiers.
        item_encoder: Label encoder used for item identifiers.
        top_k: Number of recommendations to return.

    Returns:
        A list containing the recommended item identifiers.

    Raises:
        ValueError: If ``visitor_id`` was not present during training and
            cannot be encoded by the user encoder.
    """

    user_index = user_encoder.transform(
        [visitor_id]
    )[0]

    user_tensor = torch.tensor(
        [user_index],
        dtype=torch.long,
    )

    with torch.no_grad():

        user_embedding = model.user_embedding(user_tensor)

        scores = torch.matmul(
            model.item_embedding.weight,
            user_embedding.squeeze(),
        )

    top_indices = torch.topk(
        scores,
        top_k,
    ).indices.numpy()

    return item_encoder.inverse_transform(
        top_indices
    ).tolist()


if __name__ == "__main__":
    train()