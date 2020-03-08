from fastapi import FastAPI

from enum import Enum
from typing import Optional

# app is an instance of the class FastAPI
app = FastAPI()

class ModelName(str,Enum):
    alexnet = 'alexnet'
    resnet = 'resnet'
    lenet = 'lenet'

# '/' is the path, it is the last part of the URL starting from the first /
# get is one of the 4 operations, POST, GET, PUT and DELETE
# POST: create data
# GET: read data
# PUT: update data
# DELETE: delete data
# root function will be called whenever FastAPI receives a request to the URL '/' using GET
@app.get('/')
async def root():
    return {"message":"Hello World"}

# predefined available parameters
@app.get('/model/{model_name}')
async def get_model(model_name:ModelName):
    if model_name == ModelName.alexnet:
        return {'Model_name':model_name,'message':'Deep learning FTW'}

@app.get('/files/{file_path:path}')
async def read_user_me(file_path:str):
    return {'file_pathhahaha':file_path}

# query parameters
# http://127.0.0.1:8000/items/3?skip=0&limit=1....3:item_id
fake_items = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
@app.get('/items/{item_id}/users/{user_id}/')
async def read_item2(item_id: Optional[int]=None, user_id: Optional[int]=None, skip: int = 0, limit: int = 10):
    return fake_items[skip:skip+limit], item_id, user_id
