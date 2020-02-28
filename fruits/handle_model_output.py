import cv2
import numpy as np
from PIL import Image
# import torch

def handle_output(output):
    '''
    Handles the output of the asl recognition model.
    Returns one integer = prob: the argmax of softmax output.
    '''
    # model_pred = np.argmax(output.flatten())  
    # print(model_pred)
    print(output)
    # print(max(output.flatten()))
    print( np.argmax(output.flatten()) )
    return np.argmax(output.flatten()) 
    
def crop_center(img,cropx,cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = input_image.resize((224, 224),Image.BILINEAR)
    img = np.array(image, dtype=np.float32)
    print(img.shape)
    img = img.transpose((2,0,1))
    img = img/255
    img = (img - 0.5)/0.5
    
    print('Normalized')
    print(img[0][0][:50])
    img = img.reshape(1,3, height, width)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print('Pre-processed image shape is ' + str(im_rgb.shape))
    return im_rgb