from pathlib import Path
import cv2
from constants import *

DATA_DIR = Path('data/')

def resize_images(size=(224,224)):
    for f in (DATA_DIR / 'birds').rglob('*.jpg'):
