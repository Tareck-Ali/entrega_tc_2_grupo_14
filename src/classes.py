from configs.settings import settings
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


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
        embedding_dim: int = settings.embedding_dim,
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
