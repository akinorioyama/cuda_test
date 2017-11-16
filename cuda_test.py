import numpy as np
import numba as cuda
import time

def create_matrix(bit_length):
    bit_matrix = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.byte)
    exp1 = 0
    for i in range(1,bit_length):
        exp1 = pow(2,(i+1))
        li2 = bit_matrix
        li3 = np.c_[np.zeros(((exp1),1), dtype=np.byte), li2]
        li4 = np.c_[np.ones(((exp1),1), dtype=np.byte), bit_matrix]
        li5 = np.r_[li3,li4]
        bit_matrix = li5
    print(bit_matrix)
    print(exp1)
    return bit_matrix

def dot_mul(li,numbers):
    start = time.time()
    dot_product = np.dot(np.array(li), np.array(numbers))
    elapsed_time = time.time() - start
    print ("np.dot    elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return dot_product

@cuda.jit
def Num_Dot(a, b):
    start = time.time()
    c = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i][j] += a[i][k] * b[k][j]

    elapsed_time = time.time() - start
    print ("CUDA      elapsed_time:{0}".format(elapsed_time) + "[sec]")

def NonCUDA_Num_Dot(a, b):
    start = time.time()
    c = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i][j] += a[i][k] * b[k][j]

    elapsed_time = time.time() - start
    print ("Non CUDA elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':

    bit_length = 8 #25
    bit_matrix = create_matrix(bit_length)
    numbers = np.zeros((bit_length+1, 1), dtype=np.float32)
    for i in range(bit_length+1):
        numbers[i,0] = bit_length - i +1

    print("shape", bit_matrix.shape)
    print("shape(x)", bit_matrix.shape[0]*bit_matrix.shape[1])
    print("numbers shape", numbers.shape)
    dot_product = dot_mul(bit_matrix,numbers)
    Num_Dot(bit_matrix,numbers)
    NonCUDA_Num_Dot(bit_matrix,numbers)
