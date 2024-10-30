import pandas as pd
from src.data.db_helpers import Database

db = Database()
db.connect()
df: pd.DataFrame = pd.read_csv("/home/tu/tu_tu/tu_zxonr37/master_thesis/03_Codebase/80160.csv")
db.insert_data(table="t_responses", data=df, updated_at=True)
db.disconnect()