from configs.settings import settings
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Dataset do PyTorch para registros de interações entre usuários e itens.
    
    Cada amostra é composta por um índice de usuário, um índice de item
    e a respectiva avaliação de interação utilizada como alvo de treinamento.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Inicializa o dataset de interações.
        
        Args:
            dataframe:
                DataFrame contendo os índices codificados dos usuários,
                dos itens e as avaliações de interação. O DataFrame
                deve conter as colunas ``user_idx``, ``item_idx`` e
                ``rating``.
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
        """
        Retorna o número de amostras de interação.
        
        Returns:
            Número de amostras no dataset.
        """
        return len(self.users)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Recupera uma única amostra de interação.
        
        Args:
            index: Posição da amostra a ser recuperada.
        
        Returns:
            Uma tupla contendo o índice do usuário,
            o índice do item e a avaliação alvo.
        """

        return (
            self.users[index],
            self.items[index],
            self.targets[index],
        )


class EmbeddingRecommender(nn.Module):
    """
    Sistema de recomendação baseado em fatoração de matrizes usando
    camadas de embedding.
    
    O modelo aprende representações latentes para usuários e itens.
    A pontuação de interação prevista é calculada como o produto
    escalar entre os embeddings correspondentes do usuário e do item.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = settings.embedding_dim,
    ) -> None:
        """
        Inicializa o modelo de recomendação.
        
        Args:
            num_users: Número de usuários únicos.
            num_items: Número de itens únicos.
            embedding_dim: Dimensão dos vetores de embedding latentes.
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
        """
        Calcula as pontuações previstas de interação.
        
        Args:
            users: Tensor contendo os índices dos usuários.
            items: Tensor contendo os índices dos itens.
        
        Returns: Tensor contendo a pontuação de interação
        prevista para cada par usuário-item.
        """

        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)

        return (user_vectors * item_vectors).sum(dim=1)
