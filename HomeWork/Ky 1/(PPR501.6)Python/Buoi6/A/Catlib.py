
from Animallib import Animal       
# Lớp Cat mở rộng (extends) từ lớp Animal.
class Cat (Animal): 
    def __init__(self, name, age, height):
        # Gọi tới constructor của lớp cha (Animal)
        # để gán giá trị vào thuộc tính 'name' của lớp cha.
        super().__init__(name)
        self.age = age 
        self.height = height
    # Ghi đè (override) phương thức cùng tên của lớp cha.
    def showInfo(self):
        print ("I'm " + self.name)
        print (" age " + str(self.age))
        print (" height " + str(self.height))
