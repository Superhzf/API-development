from fastapi import FastAPI,Query

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

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


# Request body
# If you use dict would also work but it is not able to do auto-completion and error checks for incorrect types and
# operations
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None


@app.post('/item_request/{item_id}')
async def create_item(item_id: int,item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price+item.tax
        item_dict.update({'price_with_tax':price_with_tax})
    return item_dict, item_id


# Query parameters and string validation
# ...: applies to q making it required, if you want q to be optional,
# max_length: applies to q making it less than 50
# regex: applies to q making it equal to fixedquery
@app.get('/item_query/')
async def read_item3(q: str = Query(..., max_length=50, regex='^fixedquery$', alias='item-query',
                                    description='the description of the parameter')):
    return q

# http://localhost:8000/item_list/?q=foo&q=bar
@app.get('/item_list/')
async def read_item3(q: List[str] = Query(['huzefu','beixuanqin'])):
    return q

