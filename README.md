# Home Credit default prediction service

- [General Description](#general-description)
- [Initial Setup](#initial-setup)
- [FIRE Commands](#fire-commands)
- [Core Technologies Used](#core-technologies-used)


# General Description
Home Credit is an international non-bank financial institution. It focuses on lending primarily
to people with little or no credit history. [Wiki page for Home Credit](https://en.wikipedia.org/wiki/Home_Credit)

This service aims to help Home Credit predict applicants' repayment abilities. The data
comes from [kaggle](https://www.kaggle.com/c/home-credit-default-risk/data). The basic
logic is as follows:
![flowchhart](images/hc_flow.png)
Please note that, this service would have the best performance on returning customers, so in reality the 
main service should call this service if and only if customers are already in the data warehouse and have
business with Home Credit.

The data set mainly includes 6 parts:
* Basic application information
* Bureau data
* Point of sale (POS) balance: POS (a.k.a consumer finance) is one of Home Credit products
* Credit card balance: credit card (a.k.a line of credit) is one of Home Credit products
* Previous application data
* Installment payments: This include installment payments of all Home Credit products

# Initial Setup
The following are required to work on the Home Credit default prediction service:
* Download and unzip data from [kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and
put all data into folder `VirtualDataWarehouse`
* Follow the instructions in folder `model-retraining` to do virtual ETL and train the model. Please note
that, the ETL part should be done in the data warehouse instead of here in reality.
* Python 3.7.3
* Poetry
* [Poetry](https://github.com/sdispater/poetry)
    * `curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`
* [Docker](https://docs.docker.com/v17.12/docker-for-mac/install/)
* Run `./fire rebuild` to build the service locally
* Run `./fire alembic upgrade` to run database migrations locally
* To make sure that everything is working, run `./go test all`

# FIRE Commands
In order to improve the development efficiency, a collection of FIRE commands are introduced:
```
usage: ./fire [-v] actions

Fire script for the default prediction service

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         run in verbose mode

actions:
  {rebuild,start,nuke_image,nuke_container,stop,shell,alembic,test}
    rebuild             Rebuild the container using docker-compose: docker-
                        compose build.
    start               Start the dev container using docker-compose: docker-
                        compose up dev
    nuke_image          Removes all docker images, even the ones unrelated to
                        this project
    nuke_container      Removes all docker containers, even the ones unrelated
                        to this project, no matter they are stopped or
                        running.
    stop                Stop all running containers, even the ones unrelated
                        to this project
    shell               Open a terminal session in this application's
                        development container.
    alembic             Run an alembic migration, you should use the parameter
                        -o with value upgrade/downgrade
    test                Run the unit test in the dev container

```
# Core Technologies Used
This is a list of core packages and frameworks that a developer needs to know in order to be able to work with this 
service and actively maintain or add features.

* [FastApi](https://github.com/tiangolo/fastapi):  API framework.
* [Alembic](https://alembic.sqlalchemy.org/en/latest/): Database migration tool.
* [SQLAlchemy](https://www.sqlalchemy.org/): Database ORM tool.
* [Pytest](https://pytest.org/en/latest/): Testing framework
* [Poetry](https://github.com/sdispater/poetry): Dependency management tool
* [Numpy](https://numpy.org/): Numerical computing package
* [Pandas](https://pandas.pydata.org/): Data manipulation package
* [PyDantic](https://pydantic-docs.helpmanual.io/): Case classes and validation tool
* [python-json-logger](https://github.com/madzak/python-json-logger): Logging library
