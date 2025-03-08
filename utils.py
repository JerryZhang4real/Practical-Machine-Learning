# utils.py
import datetime
from config import CONFIG_INFO

def log_performance(accuracy, reg_loss):
    """
    Appends the model's performance along with configuration information and a timestamp to a log file.
    
    Args:
        accuracy (float): The model's validation accuracy.
    """
    with open('performance.txt', 'a') as f:
        f.write(f"{datetime.datetime.now()}\n")
        f"Validation Accuracy: {accuracy*100:.2f}% + reg_loss: {reg_loss:.4f}\n"
        f.write(CONFIG_INFO)