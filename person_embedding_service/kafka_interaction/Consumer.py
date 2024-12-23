import asyncio
import io
import json
import os
import time
from PIL import Image
from aiokafka import AIOKafkaConsumer

from v3.partitioneddetectionbatch import PartitionedDetectionBatch

from person_embedding_service.embeding_store.EmbedingStore import EmbeddingStore
from person_embedding_service.redis_manager.RedisManager import  RedisManager
from .Producer import KafkaProducerService
from ..model.Osnet import OSNetEmbedder


class KafkaConsumerService:
    def __init__(self, bootstrap_servers='192.168.111.131:9092', topic='person_detected_response'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None  
        self.embedder = OSNetEmbedder()
        self.producer = KafkaProducerService()
        self.embedding_store = EmbeddingStore()
        self.frames_metadata_manager_client = RedisManager(db=1)
        self.persons_metadata_manager_client = RedisManager(db=2)
        self.frames_data_manager_client = RedisManager(db=0)

    async def crop(self, image_bytes, xmin, ymin, xmax, ymax):
        """Private async method to crop an image based on given coordinates."""
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes))
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        return cropped_img
    
    async def crop_images(self, frame_data, bboxs):
        images = []
        for i, bbox in enumerate(bboxs):
            for person in bbox:
                image = await self.crop(frame_data[i], person.xmin, person.ymin, person.xmax, person.ymax)
                images.append(image)
        return images
    
    async def start(self):
        # Initialize the consumer
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x:  json.loads(x.decode('utf-8'))
        )
        await self.consumer.start()

        # Start the producer
        await self.producer.start()

        try:
            await self.consume_messages()
        finally:
            await self.consumer.stop()
            await self.producer.close()

    async def consume_messages(self):
        async for message in self.consumer:
            print("There is a message")
            start_time = time.time()
            request = [PartitionedDetectionBatch(**r) for r in message.value]

            bboxs = []
            frames_completed = []
            frames_keys = []
            update_keys_list = []

            for detection in request:
                key = detection.frame_key
                if detection.partition_number == detection.total_partitions:
                    frames_completed.append(f'metadata:{key}')
                bbox = detection.person_bbox
                update_keys = detection.person_keys

                bboxs.append(bbox)
                frames_keys.append(key)
                update_keys_list.extend(update_keys)

            frame_data = self.frames_data_manager_client.get_many(frames_keys)
            images = await self.crop_images(frame_data, bboxs)
            start_time_inf = time.time()
            vectors = await self.embedder.get_embeddings(images)
            end_time_inf = time.time()
            inferance_time = end_time_inf - start_time_inf
            print(f"Time taken to inferance : {inferance_time} seconds")

            embedding_keys = self.embedding_store.save_to_milvus(vectors)
            
            self.persons_metadata_manager_client.update_by_field_many(update_keys_list, "embedding_key", embedding_keys)

            if frames_completed:
                self.frames_metadata_manager_client.update_by_field_many(frames_completed, "embeding_complete", ["True" for _ in frames_completed])
                results = self.frames_metadata_manager_client.get_values_of_field_many(frames_completed, "keypoint_complete")
                print("done")
                end_time = time.time()
                time_elapsed = end_time - start_time
                print(f"Time taken to execute the block of code: {time_elapsed} seconds")
                for i, result in enumerate(results):
                    if result == "True" :
                        await self.producer.trigger_tracker(frames_completed[i])
