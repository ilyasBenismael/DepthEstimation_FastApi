import torch
from fastapi import FastAPI, WebSocket
import websockets
import asyncio
import cv2
import numpy as np


# # Function to send image to YOLO and receive results
# async def send_to_yolo(image_bytes, yolo_websocket, flutter_websocket):
#     try:
#         # Send the image bytes to the YOLO WebSocket server
#         await yolo_websocket.send(image_bytes)

#         # Receive the processed data from YOLO
#         yolo_image_bytes = await yolo_websocket.receive_bytes()

#         # Send the YOLO results back to the Flutter client
#         await flutter_websocket.send(yolo_image_bytes)

#     except Exception as e:
#         print(f"Error in YOLO processing: {e}")
#         await flutter_websocket.send_text(f"Error in YOLO processing: {str(e)}")


# Load MiDaS model (CPU inference)
midasModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
device = torch.device("cpu")
midasModel.to(device)
midasModel.eval()

# Create FastAPI app
myapp = FastAPI()


@myapp.websocket("/midas")
async def websocket_endpoint(flutter_websocket: WebSocket):
    await flutter_websocket.accept()
    print("flutterToMidasDone")

    # Establish a persistent connection to YOLO
    yolo_websocket = None
    try:
        # Establish the connection to YOLO once
        yolo_websocket = await websockets.connect("wss://safedrivefastapi-production.up.railway.app/yolo")

        while True:
            # Receive the image bytes from the Flutter client
            image_bytes = await flutter_websocket.receive_bytes()
            print("imageFromFlutterReceived")

            await yolo_websocket.send("hey ilyas")
            print("text sent to yolo")

            lastResp = await yolo_websocket.recv()
            print("lastresponse from yolo received")

            await flutter_websocket.send(lastResp)





################# send text to yolo


            # Process the image with YOLO using the persistent connection
            # await send_to_yolo(image_bytes, yolo_websocket, flutter_websocket)

    except Exception as e:
        await flutter_websocket.send_text(f"Error akhoyaaw : {str(e)}")
        print(f"Error akhooooyaaa: {e}")



            # # Convert bytes to a NumPy array
            # nparr = np.frombuffer(image_bytes, np.uint8)
            # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # # Convert img to RGB
            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # # prepare the img via transform and put it cpu so it can be ready for midas process
            # midas_input_img = transform(imgRGB).to(device)         
            
            # # maintain the non-gradient context when calling the prediciton (cause we in inference)
            # with torch.no_grad():          
            
            #     #getting the prediction
            #     midasTensor = midasModel(midas_input_img)          
           
            #     #Resize the prediction to match the input image size before transforms
            #     midasTensor = torch.nn.functional.interpolate(
            #     midasTensor.unsqueeze(1),
            #     size=imgRGB.shape[:2],
            #     mode="bicubic",
            #     align_corners=False,
            #     ).squeeze()
        
            # #Convert the prediction to a NumPy array cause it's easy to handle better than a pytorch tensor
            # midasNumpy = midasTensor.cpu().numpy()

            # # Normalize the depth map to 0-255, and make of type uint8 se we can turn it to jpeg format later
            # midasNumpy = (midasNumpy - np.nanmin(midasNumpy)) / (np.nanmax(midasNumpy) - np.nanmin(midasNumpy)) * 255.0
            # midasNumpy = np.clip(midasNumpy, 0, 255).astype(np.uint8)  # Convert to uint8

            # # turn the numpy img to bytes
            # img_buffer = BytesIO()
            # Image.fromarray(midasNumpy).save(img_buffer, format='JPEG')
            # rendered_image_bytes = img_buffer.getvalue()

            # # send image bytes back via websocket
            # await flutter_websocket.send_bytes(rendered_image_bytes)

