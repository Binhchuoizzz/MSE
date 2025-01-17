
# Một lớp mô phỏng một hình chữ nhật.
class Rectangle :
    'This is Rectangle class'
    # Một phương thức được sử dụng để tạo đối tượng (Contructor).
    def __init__(self, width, height):
        self.width= width
        self.height = height
    def getWidth(self):        
        return self.width
    def getHeight(self):        
        return self.height
    # Phương thức tính diện tích.
    def getArea(self):
        return self.width * self.height
