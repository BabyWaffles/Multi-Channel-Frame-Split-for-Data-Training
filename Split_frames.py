import cv2
import numpy as np
import os
import logging
import av
import threading
import asyncio

current_frame_count = 0

def new_directory(folder_name):
    try:
        current_directory = os.getcwd()
        join_dir = os.path.join(current_directory, folder_name)
        if not os.path.exists(join_dir):
            os.makedirs(join_dir)
            logging.info(f"Processed img directory {folder_name} created, in {current_directory}")
            return join_dir
        else:
            logging.info("Directory already exists")
            return join_dir
    except os.error as e:
        logging.error(f"Error creating directory: {e}")
        return None
    except Exception as e:
        logging.error(f"Recursively returned NULL w/ err: {e}")
        return "NULL"

data_folder = f"{os.getcwd()}\cvat-dataInference"
#processed_path = f"{data_folder}\processed_imgs" #f"{new_directory("cvat-dataInference\processed_imgs")}"

class deepsplitting:
    def deepSplit_processed(frame, current_frame_count, processed_path):
        frames= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        current_frame_queue = frames.copy()
        b,g,r = cv2.split(current_frame_queue)
        #convert to gray
        gray = cv2.cvtColor(current_frame_queue, cv2.COLOR_BGR2GRAY)
        file_name=f"split_frame_{current_frame_count}"
        file_write(processed_path, file_name, r,g,b,gray, frame)
        print(f"Frame {current_frame_count} processed successfully")


class threadings: 
    async def worker_threads(data_folder, current_frame_count):
        threads = [6]
        print(f"Starting worker thread with: {threads} threads")
        for i in range(1, 10):
            t = threading.Thread(target=call_class_deepsplit, args=(data_folder, current_frame_count))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print("Threads completed successfully")

def file_write(path, file_name, r,g,b,gray, normal):
    with open(path + file_name, 'w') as f:
        cv2.imwrite(f"{path}/{file_name}_r.jpg", r)
        cv2.imwrite(f"{path}/{file_name}_g.jpg", g)
        cv2.imwrite(f"{path}/{file_name}_b.jpg", b)
        cv2.imwrite(f"{path}/{file_name}_gray.jpg", gray)
        cv2.imwrite(f"{path}/{file_name}_normal.jpg", normal)
        logging.info(f"Frames written successfully {file_name}")
        f.close()

def call_class_deepsplit(folder_path, current_frame_count):
    items = os.listdir(folder_path)
    files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]
    logging.info(f"Files in directory: {files}")
    processed_path = new_directory(f"{folder_path}\processed_imgs")
    print(f"Processed path: {processed_path}")
    for file in files:
        if file.endswith(".jpg"):
            logging.info(f"Processing file: {file}")
            img = cv2.imread(f"{folder_path}/{file}")   
            deepsplitting.deepSplit_processed(img, current_frame_count, processed_path)
            current_frame_count += 1
        elif file.endswith(".mp4"):
            logging.info(f"Processing video: {file}")
            cap = cv2.VideoCapture(f"{folder_path}/{file}")
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    deepsplitting.deepSplit_processed(frame, current_frame_count, processed_path)
                    current_frame_count += 1
            cap.release()
            cv2.destroyAllWindows()

#call_class_deepsplit(data_folder, current_frame_count)
loop = asyncio.get_event_loop().run_until_complete(threadings.worker_threads(data_folder, current_frame_count))
