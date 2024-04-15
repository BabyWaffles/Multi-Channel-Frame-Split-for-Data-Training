from cv2 import cvtColor, COLOR_RGB2BGR, split, COLOR_BGR2GRAY, imwrite, imread, VideoCapture, destroyAllWindows
from os import makedirs, getcwd, path, listdir
from logging import info, error
from time import time

print("Initializing Frame Splitter...")
start = time()

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
        self.red = None
        self.green = None
        self.blue = None
        self.gray = None
        self.file_name = None

    def deepSplit_processed(self):
        try:
            self.frame = cvtColor(self.frame, COLOR_RGB2BGR)
            self.blue, self.green, self.red = split(self.frame)
            self.gray = cvtColor(self.frame, COLOR_BGR2GRAY)
            self.file_name= f"split_frame_{self.current_frame_count}"
            self.file_processing()
            print(f"Frame {self.current_frame_count} processed successfully")

        except Exception as e:
            error(f"Error processing frame {self.current_frame_count}: {e}")
            #raise processingNullFrames(f"Error processing frame {self.current_frame_count}: {e}")
    
    def file_processing(self):
        with open(self.processed_path + self.file_name, 'w') as file:
            imwrite(f"{self.processed_path}/{self.file_name}_r.jpg", self.red)
            imwrite(f"{self.processed_path}/{self.file_name}_g.jpg", self.green)
            imwrite(f"{self.processed_path}/{self.file_name}_b.jpg", self.blue)
            imwrite(f"{self.processed_path}/{self.file_name}_gray.jpg", self.gray)
            imwrite(f"{self.processed_path}/{self.file_name}_normal.jpg", self.frame)
            info(f"Frame {self.current_frame_count} processed successfully")
            file.close()

end = time()
print(f"Frame Splitter initialized successfully with {end-start}ms!")

class lastly:
    def __init__(self, folder_path, global_fcount):
        self.path = folder_path
        self.frame_count = global_fcount
        self.files = None
        self.file = None
        self.current_files = None
        self.items = None
        self.process = None

    def execute(self):
        self.files = listdir(self.path)
        self.current_files = [self.items for self.items in self.files if path.isfile(path.join(self.path, self.items))]
        info(f"Listed files in current directory: {self.current_files}")
        self.process = GetCWD(f"{self.path}\processed_imgs").newdir()
        print(f"Successfully located processing path: {self.process}")
        for self.file in self.files:
            if self.file.endswith((".jpg", ".png", ".jpeg", ".tiff", ".bmp")):
                info(f"Processing file: {self.file}")
                img = imread(f"{self.path}/{self.file}")   
                deepsplit(img, self.frame_count, self.process).deepSplit_processed()
                self.process.up_framecount()
            elif self.file.endswith((".avi", ".mp4", ".mov", ".flv")):
                info(f"Processing video: {self.file}")
                cap = VideoCapture(f"{self.path}/{self.file}")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        deepsplit(frame, self.frame_count, self.process).deepSplit_processed()
                        self.frame_count.up_framecount()
                    else:
                        break
                cap.release()
                destroyAllWindows()

""" def lastly(folder_path, current_frame_count):
    items = listdir(folder_path)
    files = [item for item in items if path.isfile(path.join(folder_path, item))]
    info(f"Files in directory: {files}")
    processed_path = GetCWD(f"{folder_path}\processed_imgs").newdir()
    print(f"Processed path: {processed_path}")
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg", ".tiff", ".bmp")):

            info(f"Processing file: {file}")

            img = imread(f"{folder_path}/{file}")

            deepsplit(img, current_frame_count, processed_path).deepSplit_processed()

            current_frame_count.up_framecount()
            
        elif file.endswith((".avi", ".mp4", ".mov", ".flv")):
            info(f"Processing video: {file}")
            cap = VideoCapture(f"{folder_path}/{file}")
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    deepsplit(frame, current_frame_count, processed_path).deepSplit_processed()
                    current_frame_count.up_framecount()
                else:
                    break
            cap.release()
            destroyAllWindows() """


data_path = f"{getcwd()}"
lastly(data_path, global_framecount()).execute()
