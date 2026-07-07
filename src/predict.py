from configs.settings import settings
from sklearn.preprocessing import LabelEncoder
from classes import EmbeddingRecommender
import torch

def recommend(
    model: EmbeddingRecommender,
    visitor_id: int,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    top_k: int = settings.top_k,
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