from cv2 import cvtColor, COLOR_RGB2BGR, split, COLOR_BGR2GRAY, COLOR_BGR2RGB, imwrite
from os import makedirs, getcwd, path
from logging import info, error
from threading import Thread
from asyncio import gather

class global_frame_count:
    def __init__(self, framecount):
        return self.global_frame_count + framecount


class systemRecurrsiveNull:
    pass

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
                return 0x0811
            else:
                return self.joindirw
        except error as oserr:
            oserr(f"Error creating directory: {oserr}")	
            return None
        except systemRecurrsiveNull as srn:
            raise systemRecurrsiveNull(f"Recursively returned NULL w/ err: {srn}")	

