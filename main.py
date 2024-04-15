import pickle
import uvicorn
import pandas as pd
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# Inicia API
app = FastAPI()

# Carrega modelo
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Cria página inicial
@app.get('/')
def home():
    return 'Welcome to the Customer Service Prediction App!'

# Classifica custo (consumo do modelo)
@app.get('/predict')
def predict(
    days_since_last_SE_GI: int = -1,
    gw_g8: float = -14.95,
    to_l5_l20: float = 0.2,
    ini_bon_g10: float = 0.,
    succ_dep_g10: float = 60.,
    SE_GI_total_70days: int = 0,
    days_g2: int = 2,
    gw_g10: float = 24.64,
    GOC_to_g9: float = 0.,
    turnover_last_20days: float = 1553.36,
    succ_dep_cnt_g9: int = 0,
    SE_GI_max_datediff: int = -1
    ):
    df_input = pd.DataFrame([dict(
        days_since_last_SE_GI = days_since_last_SE_GI,
        gw_g8 = gw_g8,
        to_l5_l20 = to_l5_l20,
        ini_bon_g10 = ini_bon_g10,
        succ_dep_g10 = succ_dep_g10,
        SE_GI_total_70days = SE_GI_total_70days,
        days_g2 = days_g2,
        gw_g10 = gw_g10,
        GOC_to_g9 = GOC_to_g9,
        turnover_last_20days = turnover_last_20days,
        succ_dep_cnt_g9 = succ_dep_cnt_g9,
        SE_GI_max_datediff = SE_GI_max_datediff
    )])
    output = model.predict(df_input).tolist()[0]
    return output

class Customer(BaseModel):

    days_since_last_SE_GI: int = -1
    gw_g8: float = -14.95
    to_l5_l20: float = 0.2
    ini_bon_g10: float = 0.
    succ_dep_g10: float = 60.
    SE_GI_total_70days: int = 0
    days_g2: int = 2
    gw_g10: float = 24.64
    GOC_to_g9: float = 0.
    turnover_last_20days: float = 1553.36
    succ_dep_cnt_g9: int = 0
    SE_GI_max_datediff: int = -1

    class Config:
        schema_extra = {
            'example': {
                'days_since_last_SE_GI': -1,
                'gw_g8': -14.95,
                'to_l5_l20': 0.2,
                'ini_bon_g10': 0.,
                'succ_dep_g10': 60.,
                'SE_GI_total_70days': 0,
                'days_g2': 2,
                'gw_g10': 24.64,
                'GOC_to_g9': 0.,
                'turnover_last_20days': 1553.36,
                'succ_dep_cnt_g9': 0,
                'SE_GI_max_datediff': -1
            }
        }

# @app.post('/predict_customer_data')
# def predict(data: Customer):
#     df_input = pd.DataFrame([data.dict()])
#     output = model.predict(df_input).tolist()[0]
#     return output

class CustomerList(BaseModel):
    data: List[Customer]

@app.post('/batch_prediction')
def predict(data: CustomerList):
    df_input = pd.DataFrame(data.dict()['data'])
    output = model.predict(df_input).tolist()
    return output

# Executa API
if __name__ == '__main__':
    uvicorn.run(app)
