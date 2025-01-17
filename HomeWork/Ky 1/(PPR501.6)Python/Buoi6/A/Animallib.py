
class Animal :  
    # Constructor
    def __init__(self, name):
        # Lớp Animal có 1 thuộc tính (attribute): 'name'.
        self.name= name 
    # Phương thức (method):
    def showInfo(self):
        print ("I'm " + self.name)
    # Phương thức (method):
    def move(self):
        print ("moving ...")
