from fastapi import FastAPI, Query, Path, Body, Cookie, Header, Form, File, UploadFile, HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler,request_validation_exception_handler

from enum import Enum
from typing import Optional, List, Set,Dict, Union
from pydantic import BaseModel, Field, HttpUrl,EmailStr

# The function parameters will be recognized as follows:
#
# If the parameter is also declared in the path, it will be used as a path parameter.
# If the parameter is of a singular type (like int, float, str, bool, etc) it will be interpreted as a query parameter.
# If the parameter is declared to be of the type of a Pydantic model or a list of a type of Pydantic models,
# it will be interpreted as a request body.

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
    description: str = Field(None,title='the description of this field',max_length=10)
    price: float=Field(...,gt=0)
    tax: float=None

# Body will make importance from a query parameter to a body parameter
@app.post('/item_request/{item_id}')
async def create_item(item_id: int, item: Item, importance: int = Body(...)):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price+item.tax
        item_dict.update({'price_with_tax':price_with_tax})
    return item_dict, item_id,importance


# Query parameters and string validation
# ...: applies to q making it required, if you want q to be optional,
# max_length: applies to q making it less than 50
# regex: applies to q making it equal to fixedquery
@app.get('/item_query/')
async def read_item3(q: str = Query(..., max_length=50, regex='^fixedquery$', alias='item-query',
                                    description='the description of the parameter')):
    return q

# http://127.0.0.1:8000/item_list/3/?q=huzefu&q=beixuan
# A path parameter is always required, so it should always look like Path(...)
# ge: greater than or equal to
# gt: greater than
# le: less than or equal to
@app.get('/item_list/{item_id}/')
async def read_item3(item_id: int = Path(..., description='the description of the path parameters', ge=1),
                     q: List[str] = Query(['huzefu', 'beixuanqin'])):
    return q, item_id


# Body nested models
class Image(BaseModel):
    url:HttpUrl
    name:str

class Item(BaseModel):
    name:str
    description:str=None
    price:float
    tax:float
    tags:Set[str] = set()
    image:Image = None


# In this case, weights should be a request body, it is not a query parameter
@app.post('/index_weights/')
async def create_index_weights(weights: Dict[int, str],weights2:int):
    return weights,weights2

# Cookie and Header parameters
@app.get('/items_cookie/')
async def read_cookie(ads_id:str = Cookie(None),user_agent=Header(None),x_token:List[str]=Header(None)):
    return {"ads_id":ads_id},{'user-agent':user_agent},{'X-Token values':x_token}


# Response model
class UserBase(BaseModel):
    username:str
    email:EmailStr
    fullname:str=None

# Using UserBase model can help reduce duplicates
class UserIn(UserBase):
    password:str


class UserOut(UserBase):
    pass


class UserInDB(UserBase):
    hashed_password:str


# this way, user password will not be responded
# response_model_exclude_unset=True means it only returns attributes in UserOut that are filled out
@app.post('/items_response/', response_model=UserOut,response_model_exclude_unset=True)
async def create_item_response(user: UserIn):
    return user

# Etra models
# For example, for user models,
# the input model needs to be able to have a password
# the output model should not have a password
# the database model would probably need to have a hashed password
def fake_password_hasher(raw_password:str):
    return "supersecret"+raw_password


def fake_save_user(user_in:UserIn):
    hashed_password=fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.dict(),hashed_password=hashed_password)
    print ("User saved! ..not really!")
    return user_in_db

# Union means that the returning type could be various types
@app.post('/user_extra_models/',response_model=Union[UserOut])
async def creat_user(user_in:UserIn):
    user_saved = fake_save_user(user_in)
    return user_saved


# Form data
# It is useful if you want to receive data from field instead of JSON(Pydantic models)
@app.post('/login/',status_code=222)
async def login(username: str = Form(...),password: str = Form(...)):
    return {"username": username, "password": password}


# Request files
# bytes type will be stored in memory which only works for small files
@app.post('/files/')
async def create_file(file: bytes=File(...)):
    return {"file_size":len(file)}


# uploadfile will not consume all the memory, it will be saved to disk if too large
@app.post('/uploadfile/')
async def create_upload_file(file:UploadFile=File(...)):
    return {'filename':file.filename}

# Handling errors
items = {"foo": "The Foo Wrestlers"}
@app.get('/item_error/{item_id}')
async def read_item(item_id:str):
    if item_id not in items:
        raise HTTPException(status_code=404,detail='Here we got an exception')
    return {'item':items[item_id]}


# custom exception handler
class UnicornException(Exception):
    def __init__(self,name:str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request:Request,exc:UnicornException):
    return JSONResponse(status_code=418,content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."})


@app.get('/unicorns/{name}')
async def read_unicorn(name:str):
    if name == 'yolo':
        raise UnicornException(name=name)
    return {"unicorn_name":name}

