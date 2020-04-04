# Standard build-in library
from typing import Dict
# Related third party libraries
from pydantic import BaseModel, PositiveFloat
from fastapi import Body
# Local application/library specific imports


class OpsBase(BaseModel):
    name: str = Body(..., description='The name of the service')
    version: str = Body(..., description='The version of the service')
    pong: PositiveFloat = Body(..., description='Seconds since epoch.')


class PingOutput(OpsBase):
    virtual_mem_used_MB: PositiveFloat = Body(..., description='The amount of system memory used, in bytes')
    process_mem_used_MB: PositiveFloat = Body(..., description='The amount of memory used by this process, in bytes')


class Dependency(BaseModel):
    message: str = Body(..., description='OK if no errors, else the exception message')
    uri: str = Body(..., description='The uri used to connect to the dependency')
    connected: bool = Body(..., description='Is the service connected to dependency?')


class ConnectivityOutput(OpsBase):
    dependencies: Dict[str, Dependency]