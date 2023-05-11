# import internal libs
import argparse
import asyncio
import os
import grpc 
import grpc.aio
import cv2
import numpy as np 
import logging
import client
from pathlib import Path 
from typing import List
from farm_ng.people_detection import people_detection_pb2
from farm_ng.people_detection import people_detection_pb2_grpc 
from farm_ng.service.service_client import ClientConfig
from farm_ng.service.service_client import ServiceClient
# Must come before kivy imports
os.environ["KIVY_NO_ARGS"] = "1"

# gui configs must go before any other kivy import
from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

# kivy imports
from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from pathlib import Path 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeopleDetectionService(people_detection_pb2_grpc.PeopleDetectionServiceServicer):
    def __init__(self, models_dir: Path) -> None:
        # load the model to detect people
        self.model = cv2.dnn.readNetFromTensorflow(
            str("/data/home/amiga/apps/people-detection/src/models/frozen_inference_graph.pb"), str("/data/home/amiga/apps/people-detection/src/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
        )
        logger.info("Loaded model: %s", models_dir.absolute())

    async def detectPeople(
        self, request: people_detection_pb2.DetectPeopleRequest, context: grpc.aio.ServicerContext
    ) -> people_detection_pb2.DetectPeopleReply:
        # decode the image
        image: np.ndarray = np.frombuffer(request.image.data, dtype=request.image.dtype)
        image = np.reshape(image, (request.image.size.height, request.image.size.width, request.image.num_channels))

        logger.debug("Detecting people in image of size %s", image.shape)

        # detect people
        self.model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        detections = self.model.forward()

        logger.debug("Num detections %d", detections.shape[2])

        # create the reply
        response = people_detection_pb2.DetectPeopleReply()

        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            if class_id != 1:  # 1 is the class id for person
                continue
            confidence: float = detections[0, 0, i, 2]
            if confidence > request.config.confidence_threshold:
                x = int(detections[0, 0, i, 3] * request.image.size.width)
                y = int(detections[0, 0, i, 4] * request.image.size.height)
                w = int(detections[0, 0, i, 5] * request.image.size.width) - x
                h = int(detections[0, 0, i, 6] * request.image.size.height) - y
                response.detections.append(
                    people_detection_pb2.Detection(x=x, y=y, width=w, height=h, confidence=confidence)
                )

        logger.debug("Num detections filtered %d", len(response.detections))

        return response

class PeopleDetectionApp(App):
    def __init__(self, service_port, models_dir, config_camera,config_detector, **kwargs):
        super().__init__(**kwargs)
        self.service_port = service_port
        self.models_dir = models_dir
        self.config_camera = config_camera
        self.config_detector = config_detector
        self.async_tasks: List[asyncio.Task] = [] 

    async def serve(self):
        server = grpc.aio.server()
        people_detection_pb2_grpc.add_PeopleDetectionServiceServicer_to_server(PeopleDetectionService(self.models_dir), server)
        server.add_insecure_port(f"[::]:{self.service_port}")

        logger.info("Starting server on port %i", self.service_port)
        await server.start()

        logger.info("Server started")
        await server.wait_for_termination() 

    def build(self):
        return Builder.load_file("res/main.kv")
        
    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    async def start_server(self):  
        while self.root is None:
            await asyncio.sleep(0.01) 
        while True:
            await self.serve() 

    async def detect_people(self):  
        while self.root is None:
            await asyncio.sleep(0.01)  
        while True:
            await client.start_client(config_camera, config_detector
            )  
      

    async def app_func(self):
        async def run_wrapper() -> None:
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.async_tasks:
                task.cancel()
 
        # tasks
        self.async_tasks.append(
            asyncio.ensure_future(self.start_server())
        )
        self.async_tasks.append(
            asyncio.ensure_future(self.detect_people())
        )

        return await asyncio.gather(run_wrapper(), *self.async_tasks)

if __name__ == "__main__":

    # create the config for the service
    parser_service = argparse.ArgumentParser(prog="amiga-people-detector-service")
    parser_service.add_argument("--port", type=int, default="50091", help="The camera port.")
    parser_service.add_argument("--models-dir", type=str,  default="/data/home/amiga/apps/people-detection/src/models/", help="The path to the models directory")
    args_service = parser_service.parse_args()
    models_path = Path(args_service.models_dir).absolute()
    assert models_path.exists(), f"Models directory {models_path} does not exist."

    # create the config for the clients
    parser_client = argparse.ArgumentParser(prog="amiga-people-detector")
    parser_client.add_argument("--port-camera", type=int,default="50051", help="The camera port.")
    parser_client.add_argument("--address-camera", type=str, default="localhost", help="The camera address")
    parser_client.add_argument("--port-detector", type=int, default="50091", help="The camera port.")
    parser_client.add_argument("--address-detector", type=str, default="localhost", help="The camera address")
    parser_client.add_argument("--stream-every-n", type=int, default=5, help="Streaming frequency")
    args_client = parser_client.parse_args()
    config_camera = ClientConfig(port=args_client.port_camera, address=args_client.address_camera)
    config_camera.stream_every_n = args_client.stream_every_n
    config_detector = ClientConfig(port=args_client.port_detector, address=args_client.address_detector)
 
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            PeopleDetectionApp(args_service.port, models_path,config_camera,config_detector).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()
   
