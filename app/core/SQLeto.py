import sqlalchemy as sql
import pandas as pd
import glob
import time



class SQLeto : 

    @classmethod
    def __init__(
        self, database:str, 
        user:str, password:str) -> None:

        self.database = database
        self.user = user
        self.pwd = password
    
    @classmethod
    def create_engine(
        self)->sql.engine.Engine :

        url = f"postgresql://{self.user}:{self.pwd}@localhost:15432/{self.database}"
        engine = sql.create_engine(url)

        return engine
    
    @classmethod
    def connect_database(
        self, engine)->sql.engine.Connection :

        return engine.connect()
    
    @classmethod
    def execute_DQL(
        self, query:str)->pd.DataFrame :

        return pd.read_sql(query, self.create_engine())
    
    @classmethod
    def execute_DDL(
        self, query:str)->str:

        conn = self.connect_database(self.create_engine())
        return conn.execute(query)
    
    @classmethod
    def get_dataframes_to_upload(
        self,
        path:str) :

        list_paths = glob.glob(
            path+"*.csv")
        list_names = list()
        for file_path in list_paths :
            name_file = file_path.split('\\')[-1]
            name_dataframe = name_file.replace('.csv','')
            list_names.append(
                name_dataframe)
            
            globals()[f"df_{name_dataframe}"] = pd.read_csv(
                path+name_file)

        return list_names
    
    @classmethod
    def create_type_column(
        self,
        data:pd.DataFrame)->dict :

        type_columns = dict()
        for column in data.columns :
            if str(data[column].dtype).startswith('float') :
                type_columns[column] = 'REAL'
            elif str(data[column].dtype).startswith('int') :
                type_columns[column] = 'INTEGER'
            elif str(data[column].dtype).startswith('datetime') :
                type_columns[column] = 'DATE'
            elif str(data[column].dtype).startswith('object') :
                max_len = int(data[column].str.len().max()+50)
                if max_len > 1000 :
                    type_columns[column] = 'TEXT'
                    continue
                type_columns[column] = f'VARCHAR({max_len})'
                
        return type_columns
    
    @classmethod
    def creating_tables(
        self,
        dataframe:pd.DataFrame, 
        name_table:str,  
        primary_key:str, 
        has_fk:bool=False,
        foreign_key:str='',
        reference_table:str=''):

        query = f'CREATE TABLE {name_table}('
        schema = self.create_type_column(data=dataframe)
        for key, value in schema.items() :
            query+= f'{key} {value},'
        if has_fk :
                query+=f'FOREIGN KEY ({foreign_key}) REFERENCES {reference_table} ({foreign_key}),'
        query+= f'PRIMARY KEY ({primary_key}));'

        return query
    
    @classmethod
    def send_dataframes_to_tables(
        self,
        list_names_order:list,
        if_exists_:str)->None:

        '''
            This method receives a path of dataframes, so send it to tables in database.
        '''

        names = list_names_order
        for name in names :
            print(f"Saving table {name}")
            globals()[f"df_{name}"].to_sql(
                name, 
                self.create_engine(), 
                if_exists=if_exists_, 
                index=False)
            time.sleep(0.8)
            print(f"Table {name} saved!")
            print()