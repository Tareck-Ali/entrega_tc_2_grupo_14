from configs.settings import settings
from configs.settings import EVENT_WEIGHTS
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess() -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Pré-processa o conjunto de dados de interações.

    Carrega um DataFrame de interações brutas contendo, no mínimo, as
    seguintes colunas:
        visitorid, itemid e event.

    Os tipos de evento são convertidos em avaliações numéricas, e os
    identificadores de usuários e itens são codificados por meio de
    codificadores de rótulos para uso nas camadas de embedding.

    Returns:
        tuple: Uma tupla contendo:
            - pd.DataFrame: O DataFrame de interações processado.
            - LabelEncoder: O codificador de rótulos ajustado para os
              usuários.
            - LabelEncoder: O codificador de rótulos ajustado para os
              itens.
    """

    dataframe = pd.read_csv("data\\"+settings.csv_file_name)

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