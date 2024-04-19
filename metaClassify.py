from cv2 import cvtColor, COLOR_RGB2BGR, split, COLOR_BGR2GRAY, imwrite, imread, VideoCapture, destroyAllWindows
from os import makedirs, getcwd, path, listdir
from logging import info, error
from time import time

#CAS == Classification Alpha System
print("Initializing Classification Alpha System [ver_a0.0.1], please wait . . .")

class systemRecurrsiveNull:
    pass

class processingNullFrames:
    pass

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

class metaClassify:
    def __init__(self, path, file_name):
        self.path = path
        self.dataentry = file_name

    def classify(self):
        pass

print("Done! Listening to classifications!")

def main():
    
    writeSequence = metaClassify(f"{getcwd()}\\data", "reflect")
    writeSequence.classify()
                                 

if __name__ == "main":
    main()
