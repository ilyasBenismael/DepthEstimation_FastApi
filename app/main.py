import torch
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from PIL import Image
from io import BytesIO



####### first of all we load midas model and its transforms (the small versions)
midasModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

####### we set the device to cpu cuz we can't afford a gpu :(
device = torch.device("cpu")
midasModel.to(device)

####### we set the model to run in inference mode and not training mode
midasModel.eval()

#### that's our fast api
myapp = FastAPI()


@myapp.websocket("/midas")
async def websocket_endpoint(flutter_websocket: WebSocket):

    await flutter_websocket.accept()
    try:
        ##### i will keep the loop so we keep listening for images from my flutter app 
        while True:

            # receive the image bytes from the Flutter client
            image_bytes = await flutter_websocket.receive_bytes()

            # convert imgbytes to a numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # convert img to RGB so it can be used in the midas model
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # prepare the img via transform and put it in cpu so it can be ready for midas process
            midas_input_img = transform(imgRGB).to(device)         
            
            # maintain the non-gradient context when calling the prediciton (cause we are in inference 
            # and not training)
            with torch.no_grad():          

                # getting the depth map in form of a pytorch Tensor
                midasTensor = midasModel(midas_input_img)          
           
                # resize the prediction to match the input image size before transforms
                midasTensor = torch.nn.functional.interpolate(
                midasTensor.unsqueeze(1),
                size=imgRGB.shape[:2],
                mode="bicubic",
                align_corners=False,
                ).squeeze()
        
            # convert the prediction to a numpy array cause it's easy to handle better than a pytorch tensor
            midasNumpy = midasTensor.cpu().numpy()

            # normalize the depth map to 0-255 and make of type uint8 se we can turn it to jpeg format later
            midasNumpy = (midasNumpy - np.nanmin(midasNumpy)) / (np.nanmax(midasNumpy) - np.nanmin(midasNumpy)) * 255.0
            midasNumpy = np.clip(midasNumpy, 0, 255).astype(np.uint8) 

            # turn the numpy img to bytes se we ca send it back
            img_buffer = BytesIO()
            Image.fromarray(midasNumpy).save(img_buffer, format='JPEG')
            final_depth_image = img_buffer.getvalue()

            # send image bytes back via websocket
            await flutter_websocket.send_bytes(final_depth_image)

    except Exception as e:
        await flutter_websocket.send_text(f"Error : {str(e)}")