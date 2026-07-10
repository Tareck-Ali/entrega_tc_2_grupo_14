from configs.settings import settings
from sklearn.preprocessing import LabelEncoder
from classes import EmbeddingRecommender
import torch

def recommend(
    model: EmbeddingRecommender = torch.load("data"+settings.model_checkpoint_name),
    visitor_id: int,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    top_k: int = settings.top_k,
) -> list[int]:
    """
    Gera recomendações de itens para um usuário.

    As recomendações são produzidas calculando a similaridade entre o
    embedding do usuário e os embeddings de todos os itens e, em
    seguida, selecionando os itens com as maiores pontuações.

    Args:
        model: Modelo de recomendação treinado.
        visitor_id: Identificador original do usuário.
        user_encoder: Codificador de rótulos utilizado para os
            identificadores dos usuários.
        item_encoder: Codificador de rótulos utilizado para os
            identificadores dos itens.
        top_k: Número de recomendações a serem retornadas.

    Returns:
        Uma lista contendo os identificadores dos itens recomendados.

    Raises:
        ValueError: Se ``visitor_id`` não estava presente durante o
            treinamento e não puder ser codificado pelo codificador de
            usuários.
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
