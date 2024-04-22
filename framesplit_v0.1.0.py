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
    
class formatClassify:
    def __init__(self, basename):
        self.currentFormat_IMG = list(".jpg" or ".JPG", ".png" or ".PNG", ".jpeg" or ".JPEG", ".tiff" or ".TIFF", ".bmp" or ".BMP")
        self.currentFormat_VIDEO = list(".avi" or ".AVI", ".mp4" or ".MP4", ".mov" or ".MOV", ".flv" or ".FLV")
        self.basename = basename

    def implement(self):
        if self.basename.endswith(self.currentFormat_IMG):
            return self.basename.endswith
        if self.basename.endswith(self.currentFormat_VIDEO):
            return self.basename.endswith
        else:
            return None

class GetCWD:
    def __init__(self, folder_name) -> str :
        self.fname = folder_name
        self.dir = getcwd()
    
    def newdir(self):
        try:
            self.joindirw =  path.join(self.dir, self.fname)
            if not path.exists(self.joindirw):
                info(f"Processed img directory {self.fname} created, in {self.dir}")
                return makedirs(self.joindirw)
            if self.fname == None:
                raise systemRecurrsiveNull(f"No directory name provided!")
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
        self.file_name= f"split_frame_{self.current_frame_count}"

    def deepSplit_processed(self):
        try:
            self.frame = cvtColor(self.frame, COLOR_RGB2BGR)
            self.queue = self.frame.copy()
            self.blue, self.green, self.red = split(self.queue)
            self.gray = cvtColor(self.queue, COLOR_BGR2GRAY)
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

class metaClassify:
    def __init__(self, path, count, processed_path):
        self.path = path
        self.current_frame_count = count.framecount
        self.ppath = processed_path
        self.file_name= None
        self.data = None

    def classify(self):
        self.file_name = f"split_frame_{self.current_frame_count}"
        with open(self.path, 'r') as file:
            self.data = file.read()
            file.close()
        with open(f"{self.ppath}\{self.file_name}_r.txt", 'w') as file: 
            file.write(self.data) 
            file.close()
        with open(f"{self.ppath}\{self.file_name}_g.txt", 'w') as file: 
            file.write( self.data)
            file.close()
        with open(f"{self.ppath}\{self.file_name}_b.txt", 'w') as file: 
            file.write(self.data) 
            file.close()
        with open(f"{self.ppath}\{self.file_name}_gray.txt", 'w') as file: 
            file.write(self.data)
            file.close()
        with open(f"{self.ppath}\{self.file_name}_normal.txt", 'w') as file: 
            file.write(self.data)
            file.close()
        print(f"Frame {self.current_frame_count} inference data processed successfully")




class lastly:
    def __init__(self, folder_path, global_fcount, global_fINdex):
        self.path = folder_path
        self.frame_count = global_fcount
        self.frame_index = global_fINdex
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
            if self.file.endswith((".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG", ".tiff", ".TIFF", ".bmp", ".BMP")): # <- Bug probe inserted (".uppercase won't be detected")
                info(f"Processing file: {self.file}")
                img = imread(f"{self.path}/{self.file}")   
                deepsplit(img, self.frame_count, self.process).deepSplit_processed()
                self.frame_count.up_framecount()
            if self.file.endswith((".txt", ".TXT")):
                metaClassify(f"{self.path}/{self.file}", self.frame_index, self.process).classify()
                self.frame_index.up_framecount()
            elif self.file.endswith((".avi", ".AVI", ".mp4", ".MP4", ".mov", ".MOV", ".flv", ".FLV")):
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

def main():
    data_path = f"{getcwd()}\data"
    lastly(data_path, global_framecount(), global_framecount()).execute()

if __name__ == "__main__":
    main()
