
#Module my_math

print('--- start of my_math ---')
PI = 3.14159
E = 2.71828
message = 'My math module'

def sum(start, *numbers):
    '''Calculate the sum of unlimited number
    Params:
        start:int/float, the start sum
        *numbers:int/float, the numbers to sum up
    Return: int/float
    '''
    for x in numbers:
        start += x
    return start

def sum_range(start, stop, step=1):
    '''    Calculate the sum of intergers
    Params:
        start:int, start range number
        stop:int, stop range number
        step:int, the step between value
    Returns: int
    '''
    sum = 0
    for i in range(start, stop, step):
        sum += i
    return sum

def fact(n):
    '''Calculate the factorial of n
    Params:
        n:int
    Return: int
    '''
    p = 1
    for i in range(1, n + 1):
        p *= i
    return p

print('--- start of my_math ---')
