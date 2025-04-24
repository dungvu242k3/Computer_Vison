import json
from typing import Callable, List, Type, Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


class DataProcessor :
    def __init__(self,batch_size = 32, transform = None):
        self.batch_size = batch_size
        self.transform = transform
    
    def load_data_from_csv(self,file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    def load_data_from_ezcel(self,file_path: str) -> pd.DataFrame:
        return pd.read_exel(file_path)
    
    def load_data_from_json(self,file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path)
    
    def load_data_from_sql(self, query: str, db_url: str) -> pd.DataFrame:
        engine = create_engine(db_url)
        return pd.read_sql(query, engine)

    def load_custom_data(self, loader_func: Callable) -> Union[pd.DataFrame, List[dict], Dataset]:
        return loader_func()
    
    def convert_to_model(self, df: pd.DataFrame, model_class: Type) -> List:
        model_objects = []
        for _, row in df.iterrows():
            model_object = model_class(**row.to_dict())
            model_objects.append(model_object)
        return model_objects

    def split_data(self, data: Union[pd.DataFrame, Dataset], test_size: float = 0.2) -> tuple:
        if isinstance(data, pd.DataFrame):
            train_data, test_data = train_test_split(data, test_size=test_size)
            return train_data, test_data
        else:
            train_size = int(len(data) * (1 - test_size))
            test_size = len(data) - train_size
            train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
            return train_data, test_data

    def load_and_process_data(
        self,
        source_type: str,
        source_path: str = None,
        model_class: Type = None,
        loader_func: Callable = None,
        test_size: float = 0.2
    ) -> tuple:

        data = None
        if source_type == 'csv':
            data = self.load_data_from_csv(source_path)
        elif source_type == 'excel':
            data = self.load_data_from_excel(source_path)
        elif source_type == 'json':
            data = self.load_data_from_json(source_path)
        elif source_type == 'sql':
            data = self.load_data_from_sql(source_path, model_class)
        elif source_type == 'custom' and loader_func:
            data = self.load_custom_data(loader_func)

        if isinstance(data, pd.DataFrame) and model_class:
            data = self.convert_to_model(data, model_class)

        return self.split_data(data, test_size)

    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> TorchDataLoader:
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        