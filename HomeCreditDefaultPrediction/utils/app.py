import os


def get_app_name() -> str:
    return os.environ["APP_NAME"]


def get_app_version() -> str:
    return os.environ["APP_VERSION"]