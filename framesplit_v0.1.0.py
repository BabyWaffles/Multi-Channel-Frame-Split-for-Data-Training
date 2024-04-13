from cv2 import cvtColor, COLOR_RGB2BGR, split, COLOR_BGR2GRAY, COLOR_BGR2RGB, imwrite
from os import makedirs, getcwd, path
from logging import info, error
from threading import Thread
from asyncio import gather

class systemRecurrsiveNull:
    pass

class processingNullFrames:
    pass

class global_framecount:
    def __init__(self, framecount):
        self.framecount = framecount
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
        self.current_frame_count = current_frame_count
        self.processed_path = processed_path
        self.queue
        self.red
        self.green
        self.blue
        self.gray

    def deepSplit_processed(self):
        try:
            self.frame = cvtColor(self.frame, COLOR_RGB2BGR)

            self.queue = self.frame.copy()

            self.blue,self.green,self.red = split(self.queue)

            self.gray = cvtColor(self.queue, COLOR_BGR2GRAY)

            file_name=f"split_frame_{self.current_frame_count}"

            file_write(self.processed_path, file_name, self.red,self.green,self.blue,self.gray, self.frame)

            print(f"Frame {self.current_frame_count} processed successfully")

        except Exception as e:
            error(f"Error processing frame {self.current_frame_count}: {e}")
            raise processingNullFrames(f"Error processing frame {self.current_frame_count}: {e}")
