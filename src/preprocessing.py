import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess() -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Preprocesses the interaction dataset.

    Loads a raw interaction DataFrame containing at least the following columns:
        visitorid, itemid, and event.

    Event types are converted into numeric ratings, and user and item
    identifiers are label encoded for use in embedding layers.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed interaction DataFrame.
            - LabelEncoder: The fitted user label encoder.
            - LabelEncoder: The fitted item label encoder.
    """

    dataframe = pd.read_csv("events.csv") # settings

    dataframe["rating"] = dataframe["event"].map(EVENT_WEIGHTS) # make 2 settings

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    dataframe["user_idx"] = user_encoder.fit_transform(
        dataframe["visitorid"]
    )

    dataframe["item_idx"] = item_encoder.fit_transform(
        dataframe["itemid"]
    )

    return dataframe, user_encoder, item_encoder