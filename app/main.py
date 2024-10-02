import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np


# Load the MiDaS model version small
midasModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# Load the appropriate transforms for midas
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform 

# put the model in cpu or gpu and 
device = torch.device("cpu")  
midasModel.to(device)

# make the model ready for inference and not training
midasModel.eval()

# Create FastAPI app
myapp = FastAPI()


@myapp.websocket("/midas")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the image bytes from the client
            image_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert img to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # prepare the img via transform and put it cpu so it can be ready for midas process
            midas_input_img = transform(imgRGB).to(device)         
            
            # maintain the non-gradient context when calling the prediciton (cause we in inference)
            with torch.no_grad():          
            
                #getting the prediction
                midasTensor = midasModel(midas_input_img)          
           
                #Resize the prediction to match the input image size before transforms
                midasTensor = torch.nn.functional.interpolate(
                midasTensor.unsqueeze(1),
                size=imgRGB.shape[:2],
                mode="bicubic",
                align_corners=False,
                ).squeeze()
        
            #Convert the prediction to a NumPy array cause it's easy to handle better than a pytorch tensor
            midasNumpy = midasTensor.cpu().numpy()

            # Normalize the depth map to 0-255, and make of type uint8 se we can turn it to jpeg format later
            midasNumpy = (midasNumpy - np.nanmin(midasNumpy)) / (np.nanmax(midasNumpy) - np.nanmin(midasNumpy)) * 255.0
            midasNumpy = np.clip(midasNumpy, 0, 255).astype(np.uint8)  # Convert to uint8

            # turn the numpy img to bytes
            img_buffer = BytesIO()
            Image.fromarray(midasNumpy).save(img_buffer, format='JPEG')
            rendered_image_bytes = img_buffer.getvalue()

            # send image bytes back via websocket
            await websocket.send_bytes(rendered_image_bytes)
            

    except Exception as e:
        await websocket.send_text(e)
        print(f"Error: {e}")