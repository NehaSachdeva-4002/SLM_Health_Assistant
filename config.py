class Config:
    SECRET_KEY = 'change-me'
    DEBUG = True
    # Add other settings like model path, logging config etc.
    MODEL_PATH = 'model/final_model'
import os

# Model directory path
MODEL_PATH = os.getenv("MODEL_PATH", "./model/final_model")
