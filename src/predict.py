from configs.settings import settings
import torch

visitor_id = settings.visitor_id
top_k = settings.top_k
model = torch.load(settings.model_checkpoint_name)
user_encoder = torch.load(settings.user_encoder)
item_encoder = torch.load(settings.item_encoder)

def get_user_tensor() -> torch.tensor:
    """
    Gera as top-K recomendações de itens para o usuário atual.

    A função obtém a representação do usuário, calcula seu embedding,
    pontua todos os itens por meio do produto escalar entre o embedding do
    usuário e os embeddings dos itens, seleciona os K itens com maior
    pontuação e retorna seus identificadores originais.

    Returns:
        list[int]: Lista com os IDs dos itens recomendados, ordenados da
            maior para a menor relevância prevista.
    """
    user_index = user_encoder.transform(
        [visitor_id]
    )[0]

    user_tensor = torch.tensor(
        [user_index],
        dtype=torch.long,
    )

    return user_tensor

def recommend() -> list[int]:
    user_tensor = get_user_tensor()

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
