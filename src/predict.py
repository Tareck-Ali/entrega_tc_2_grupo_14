from configs.settings import settings
from sklearn.preprocessing import LabelEncoder
from classes import EmbeddingRecommender
import torch

#function above 20 lines!!! Remake docstring!!!

def recommend(
    visitor_id: int,
    top_k: int = settings.top_k,
) -> list[int]:
    """
    Pega o modelo e gera recomendações de itens para um usuário.

    As recomendações são produzidas calculando a similaridade entre o
    embedding do usuário e os embeddings de todos os itens e, em
    seguida, selecionando os itens com as maiores pontuações.

    Args:
        visitor_id: Identificador original do usuário.
        top_k: Número de recomendações a serem retornadas.

    Returns:
        Uma lista contendo os identificadores dos itens recomendados.

    Raises:
        ValueError: Se ``visitor_id`` não estava presente durante o
            treinamento e não puder ser codificado pelo codificador de
            usuários.
    """

    model = torch.load("data"+settings.model_checkpoint_name)
    user_encoded = load(settings.user_encoded)
    item_encoded = load(settings.item_encoded)

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
