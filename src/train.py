from configs.settings import settings
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from classes import EmbeddingRecommender, InteractionDataset
from preprocessing import preprocess
from torch.optim import Adam
from torch import nn
from pathlib import Path
import mlflow
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def prepare_data():
    """
    Prepara os carregadores de dados de treinamento e validação para o sistema de recomendação.

    Esta função:
    - Carrega e pré-processa os dados brutos de interação através de `preprocess()`
    - Codifica usuários e itens em índices inteiros
    - Divide o conjunto de dados em conjuntos de treinamento e validação
    - Encapsula as divisões em objetos `InteractionDataset`
    - Cria DataLoaders do PyTorch para agrupamento em batches

    Returns:
        tuple: Uma tupla contendo:
            - df (pd.DataFrame): Conjunto de dados completo pré-processado com índices de usuário/item codificados
            - user_encoder: Codificador usado para transformar IDs de usuários em índices
            - item_encoder: Codificador usado para transformar IDs de itens em índices
            - train_loader (DataLoader): Carregador de dados de treinamento
            - val_loader (DataLoader): Carregador de dados de validação
    """
    df, user_encoder, item_encoder = preprocess()

    train_df, val_df = train_test_split(
        df, test_size=settings.ttsplit_test_size, random_state=settings.ttsplit_random_state
    )

    train_dataset = InteractionDataset(train_df)
    val_dataset = InteractionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size)

    return df, user_encoder, item_encoder, train_loader, val_loader

def build_model(df:pd.dataframe):
    """
    Constrói o modelo de recomendação baseado em embeddings junto com o otimizador e a função de perda.

    O modelo aprende embeddings de usuários e itens para prever pontuações de interação.

    Args:
        df (pd.DataFrame): Conjunto de dados pré-processado contendo pelo menos as colunas 'user_idx' e 'item_idx'.

    Returns:
        tuple: Uma tupla contendo:
            - model (EmbeddingRecommender): Modelo de recomendação inicializado
            - optimizer (torch.optim.Optimizer): Otimizador Adam para treinamento
            - criterion (torch.nn.Module): Função de perda (MSELoss)
    """
    model = EmbeddingRecommender(
        num_users=df["user_idx"].nunique(),
        num_items=df["item_idx"].nunique(),
        embedding_dim=64,
    )

    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.MSELoss()

    return model, optimizer, criterion

def train_epoch(model:EmbeddingRecommender, loader:any, optimizer:any, criterion:any):
    """
    Treina o modelo por uma época sobre o conjunto de dados fornecido.

    Executa o forward pass, cálculo da perda, retropropagação e atualizações do otimizador.

    Args:
        model (torch.nn.Module): Modelo de recomendação
        loader (DataLoader): Carregador de dados de treinamento
        optimizer (torch.optim.Optimizer): Otimizador usado para atualização dos parâmetros
        criterion (torch.nn.Module): Função de perda

    Returns:
        float: Perda média de treinamento em todos os batches da época
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
    Avalia o modelo no conjunto de dados de validação.

    Executa inferência sem cálculo de gradientes e calcula a perda média.

    Args:
        model (torch.nn.Module): Modelo de recomendação treinado
        loader (DataLoader): Carregador de dados de validação
        criterion (torch.nn.Module): Função de perda

    Returns:
        float: Perda média de validação em todos os batches
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
    Configura e inicia uma execução de experimento no MLflow.

    Esta função:
    - Define o nome do experimento como "recommendation-system"
    - Inicia e retorna um contexto de execução do MLflow

    Returns:
        mlflow.ActiveRun: Gerenciador de contexto da execução ativa do MLflow
    """
    mlflow.set_experiment("recommendation-system")
    return mlflow.start_run()

def save_artifacts(user_encoder:LabelEncoder, item_encoder:LabelEncoder, model:EmbeddingRecommender):
    """
    Salva artefatos do modelo e os registra no MLflow.

    Esta função:
    - Cria um diretório local de artefatos caso ele não exista
    - Serializa os codificadores de usuário e item usando joblib
    - Registra os codificadores como artefatos no MLflow
    - Registra o modelo PyTorch treinado no MLflow

    Args:
        user_encoder: Codificador usado para transformação de IDs de usuários
        item_encoder: Codificador usado para transformação de IDs de itens
        model (torch.nn.Module): Modelo de recomendação treinado
    """
    Path("artifacts").mkdir(exist_ok=True)

    joblib.dump(user_encoder, settings.user_encoder)
    joblib.dump(item_encoder, settings.item_encoder)

    mlflow.log_artifact(settings.user_encoder)
    mlflow.log_artifact(settings.item_encoder)

    mlflow.pytorch.log_model(model, artifact_path="model")

def save_checkpoint(epoch:int, model:EmbeddingRecommender, optimizer:any, loss:float | any) -> None:
    checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
    }

    torch.save(checkpoint, settings.model_checkpoint_name)

def train() -> None:
    """
    Executa o pipeline completo de treinamento do sistema de recomendação.

    Esta função:
    - Prepara os carregadores de dados e codificadores
    - Constrói o modelo, o otimizador e a função de perda
    - Inicializa o rastreamento do MLflow
    - Treina o modelo por um número fixo de épocas (10)
    - Registra métricas de treinamento/validação no MLflow
    - Salva o modelo treinado e os codificadores como artefatos

    Returns:
        None
    """
    df, user_enc, item_enc, train_loader, val_loader = prepare_data()
    model, optimizer, criterion = build_model(df)

    with setup_mlflow(): # change encodings to be saved later!
        mlflow.log_params({
            "embedding_dim": settings.embedding_dim,
            "batch_size": settings.batch_size,
            "epochs": settings.epochs,
            "learning_rate": settings.learning_rate,
        })
        for epoch in range(settings.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            save_checkpoint()
        save_artifacts(user_enc, item_enc, model)
"""
def load_checkpoint() -> None:
    checkpoint = torch.load("checkpoint.pth")

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    model.train()
"""