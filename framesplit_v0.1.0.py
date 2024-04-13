from cv2 import cvtColor, COLOR_RGB2BGR, split, COLOR_BGR2GRAY, imwrite, imread, VideoCapture, destroyAllWindows
from os import makedirs, getcwd, path, listdir
from logging import info, error
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from asyncio import gather, get_event_loop

print("Initializing Frame Splitter...")

class systemRecurrsiveNull:
    pass

class processingNullFrames:
    pass

class global_framecount:
    def __init__(self):
        self.framecount = 0
    def up_framecount(self):
        self.framecount += 1
        return self.framecount

class GetCWD:
    def __init__(self, folder_name):
        self.fname = folder_name
        self.dir = getcwd()
    
    def newdir(self):
        try:
            self.joindirw =  path.join(self.dir, self.fname)
            if not path.exists(self.joindirw):
                info(f"Processed img directory {self.fname} created, in {self.dir}")
                return makedirs(self.joindirw), 
            elif self.fname == None:
                raise systemRecurrsiveNull(f"No directory name provided!")
            else:
                return self.joindirw
        except error as oserr:
            oserr(f"Error creating directory: {oserr}")	
            return None
        except systemRecurrsiveNull as srn:
            raise systemRecurrsiveNull(f"Recursively returned NULL w/ err: {srn}")
        
class deepsplit:
    def __init__(self, frame, current_frame_count, processed_path):
        self.frame = frame
        self.current_frame_count = current_frame_count.framecount
        self.processed_path = processed_path
        self.queue = None
        self.red = None
        self.green = None
        self.blue = None
        self.gray = None
        self.file_name = None
        self.threadexecutor = ThreadPoolExecutor(max_workers=5)

    async def deepSplit_processed(self):
        try:
            self.frame = cvtColor(self.frame, COLOR_RGB2BGR)
            self.queue = self.frame.copy()
            self.blue,self.green,self.red = split(self.queue)
            self.gray = cvtColor(self.queue, COLOR_BGR2GRAY)
            self.file_name=f"split_frame_{self.current_frame_count}"
            await self.file_processing()
            print(f"Frame {self.current_frame_count} processed successfully")

        except Exception as e:
            error(f"Error processing frame {self.current_frame_count}: {e}")
            raise processingNullFrames(f"Error processing frame {self.current_frame_count}: {e}")
    
    async def file_processing(self):
        with open(self.processed_path + self.file_name) as file:
            self.imwrite(f"{self.processed_path}/{self.file_name}_r.jpg", self.red)
            self.async_imwrite(f"{self.processed_path}/{self.file_name}_g.jpg", self.green)
            self.async_imwrite(f"{self.processed_path}/{self.file_name}_b.jpg", self.blue)
            self.async_imwrite(f"{self.processed_path}/{self.file_name}_gray.jpg", self.gray)
            self.async_imwrite(f"{self.processed_path}/{self.file_name}_normal.jpg", self.queue)
            info(f"Frame {self.current_frame_count} processed successfully")
            file.close()

    async def async_imwrite(self, filename, img):
        self.loop = get_event_loop()
        await self.loop.run_in_executor(self.threadexecutor, imwrite, filename, img)

print("Frame Splitter initialized successfully!")

async def lastly(folder_path, current_frame_count):
    items = listdir(folder_path)
    files = [item for item in items if path.isfile(path.join(folder_path, item))]
    info(f"Files in directory: {files}")
    processed_path = GetCWD(f"{folder_path}\processed_imgs").newdir()
    print(f"Processed path: {processed_path}")
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg", ".tiff", ".bmp")):
            info(f"Processing file: {file}")
            img = imread(f"{folder_path}/{file}")   
            await deepsplit(img, current_frame_count, processed_path).deepSplit_processed()
            current_frame_count.up_framecount()
        elif file.endswith((".avi", ".mp4", ".mov", ".flv")):
            info(f"Processing video: {file}")
            cap = VideoCapture(f"{folder_path}/{file}")
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    await deepsplit(frame, current_frame_count, processed_path).deepSplit_processed()
                    current_frame_count.up_framecount()
                else:
                    break
            cap.release()
            destroyAllWindows()

data_folder = f"{getcwd()}\cvat-dataInference"
#processed_path = f"{data_folder}\processed_imgs" #f"{GetCWD("cvat-dataInference\processed_imgs").newdir()}" #Deprecated! Integrated it into Lastly function!
framecount = global_framecount()
lastly(data_folder, framecount)
