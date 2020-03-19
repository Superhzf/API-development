from fastapi import FastAPI, Query, Path, Body, Cookie, Header, Form, File, UploadFile, HTTPException,Request, Depends,status,\
                    BackgroundTasks, Response
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler,request_validation_exception_handler
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from enum import Enum
from typing import Optional, List, Set,Dict, Union, Callable, Any
from pydantic import BaseModel, Field, HttpUrl,EmailStr
from datetime import datetime
import time

# The function parameters will be recognized as follows:
#
# If the parameter is also declared in the path, it will be used as a path parameter.
# If the parameter is of a singular type (like int, float, str, bool, etc) it will be interpreted as a query parameter.
# If the parameter is declared to be of the type of a Pydantic model or a list of a type of Pydantic models,
# it will be interpreted as a request body.

# app is an instance of the class FastAPI
app = FastAPI(title="Study how to develop REST API using FastAPI",
              description='This is a very fancy project',
              version='0.0.1')

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


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


@app.get('/unicorns/{name}',tags=['Exception-handler'], summary='handle exception', response_description='hhh')
async def read_unicorn(name:str):
    # below is the docstring which works the same as the description path parameter
    """
    Study how to handle exceptions of FastAPI

    :param name:
    :return:
    """
    if name == 'yolo':
        raise UnicornException(name=name)
    return {"unicorn_name":name}

# JSON compatible encoder
fake_db = {}


class ItemDB(BaseModel):
    title:str
    timestamp:datetime
    description:str=None


@app.get('/json_encoder/{id}')
async def save_item(id:str,item:ItemDB):
    # jsonable_encoder will transform timestamp to str which is JSON compatible
    # at the same time, ItemDB will be transformed to a python dict
    json_compitable_data = jsonable_encoder(item)
    fake_db[id] = json_compitable_data


# Body updates
class UpdateItem(BaseModel):
    name:str=None
    description:str=None
    price:float=None
    tax:float = 10.5
    tags:List[str]=[]


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}

# this way, if the item does not include tax, then the returned object will have the default value
@app.put('/item_update/{item_id}',response_model=UpdateItem)
async def update_item(item_id: str, item: UpdateItem):
    update_item_encoded = jsonable_encoder(item)
    items[item_id] = update_item_encoded
    return update_item_encoded

# with the help of patch, you can update what you want and leave the rest intact
@app.patch('/item_update/{item_id}',response_model=UpdateItem)
async def partial_update(item_id:str,item:UpdateItem):
    stored_item_data = items[item_id]
    stored_item_model = Item(**stored_item_data)
    update_data = item.dict(exclude_unset=True)
    updated_item = stored_item_model.copy(update=update_data)
    items[item_id] = jsonable_encoder(updated_item)
    return updated_item


# Dependency Injection
async def common_parameter(q:str=None,skip:int=0,limit:int=100):
    return {'q':q,'skip':skip,'limit':limit}


@app.get('/dependencies/')
async def read_items(commons:dict = Depends(common_parameter)):
    return commons


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self,q:str=None,skip:int=0,limit:int=100):
        self.q = q
        self.skip = skip
        self.limit = limit

# Depends() == Depends(CommonQueryParams)
@app.get('/class_dependencies/')
async def read_items(commons:CommonQueryParams = Depends()):
    response = {}
    if commons.q:
        response.update({'q': commons.q})
    items_get = fake_items_db[commons.skip:commons.skip+commons.limit]
    response.update({'items': items_get})
    return response


# dependencies in path operation decorators
async def verify_token(x_token:str = Header(...)):
    if x_token!='fake-super-secret-token':
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key:str=Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")

# In the case that we do not need any returned values from dependencies,
# we can put them in path parameters
@app.get('/dependencies_path/', dependencies=[Depends(verify_key),Depends(verify_token)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]


# Security
class User(BaseModel):
    username:str
    email: Optional[EmailStr]=None
    full_name:Optional[str] = None
    disabled:Optional[bool] = None


def fake_decode_token(token):
    return User(username=token+'fakedecoded',email='john@example.com',full_name='John Doe')


async def get_current_user(token:str=Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    return user


@app.get('/security/me')
async def read_items(token:str=Depends(oauth2_scheme)):
    return {'token':token}


@app.get('/users/me')
async def read_users_me(current_user:User=Depends(get_current_user)):
    return current_user


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}


def fake_hash_password(password:str):
    return 'fakehashed'+password


class UserInDB2(User):
    hashed_password:str

def get_user(db,username:str):
    if username in db:
        user_dict = db[username]
        return UserInDB2(**user_dict)

def fake_decode_token2(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db,token)
    return user


async def get_current_user(token:str=Depends(oauth2_scheme)):
    user = fake_decode_token2(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Invalid authentication credentials',
                            headers={"WWW-Authenticate": "Bearer"})
    return user


async def get_current_active_user(current_user:User=Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail='Inactive user')
    return current_user


@app.post('/token')
async def login(form_data:OAuth2PasswordRequestForm=Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail='Incorrent username')

    user = UserInDB2(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if user.hashed_password != hashed_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail='Incorrect password')
    return {"access_token": user.username, "token_type": "bearer"}


@app.get('/users/me')
async def read_users_me2(current_user:User=Depends(get_current_active_user)):
    return current_user


# Cross Origin Resource Sharing
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True, # Cookies should be supported
    allow_methods=['*'],
    allow_headers=['*']
)


# Background tasks
# this is the task function which can be async or not
def write_notification(email:str,message=''):
    with open('log.txt',mode='w') as email_file:
        content=f'notification for {email}:{message}'
        email_file.write(content)


@app.get("/send-notification/{email}")
async def send_notification(email:str,background_tasks:BackgroundTasks):
    # parameters of the task function will be put in any sequence after the function
    background_tasks.add_task(write_notification, email, message='email sent notification')
    return {'message': "Notification sent in the background"}


# Middleware
# If there are two middleware functions, they will be executed in order
@app.middleware('http')
async def second_add_process_time(request: Request, call_next: Callable) -> Any:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time + 100
    response.headers["X-Process-Time22222"] = str(process_time)
    # return response
    return Response("hahahah", status_code=500)


@app.middleware('http')
async def add_process_time(request: Request, call_next: Callable) -> Any:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time1111111"] = str(process_time)
    return Response("Test two middleware functions", status_code=500)
    # return response









if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8080)
