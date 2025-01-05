
from Animallib import Animal       
# Lớp Mouse mở rộng (extends) từ lớp Animal.
class Mouse (Animal): 
    def __init__(self, name, age, height):
        # Gọi tới Constructor của lớp cha (Animal)
        # để gán giá trị vào thuộc tính 'name' của lớp cha.
        super().__init__(name)
        self.age = age 
        self.height = height
    # Ghi đè (override) phương thức cùng tên của lớp cha.
    def showInfo(self):
        # Gọi phương thức của lớp cha.
        super().showInfo()
        print (" age " + str(self.age))
        print (" height " + str(self.height))
    # Ghi đè (override) phương thức cùng tên của lớp cha.        
    def move(self):
        print ("Mouse moving ...")
