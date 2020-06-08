# Home Credit default prediction service

- [General Description](#general-description)
- [FIRE Commands](#fire-commands)
- [Core Technologies Used](#core-technologies-used)


# General Description
Home Credit is an international non-bank financial institution. It focuses on lending primarily
to people with little or no credit history. [Wiki page for Home Credit](https://en.wikipedia.org/wiki/Home_Credit)

This production ready service aims to help Home Credit predict applicants' repayment abilities. The data
comes from [kaggle](https://www.kaggle.com/c/home-credit-default-risk). The basic
logic is as follows:
![flowchhart](images/hc_flow.png)

The data set mainly includes 6 parts:
- Basic application information
- Bureau data
- Point of sale (POS) balance
- Credit card balance
- Previous application data
- Installment payments

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
