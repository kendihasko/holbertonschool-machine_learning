'''
LINEAR ALGEBRA
'''

# Vector - array, list of numbers

my_vector = [1,2,3,4,5]

print(my_vector)


# A function that adds two vectors vec1 and vec2

def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        return None
    
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])
    
    return result

v1 = [1, 2, 3]
v2 = [4, 5, 6]
result = add_vectors(v1, v2)
print(result)

#To find the shape of a matrix, we can use the built-in len() function to get the length 
# of the outer list (which gives the number of rows), and then get the length of one 
# of the inner lists (which gives the number of columns)

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Get the number of rows
num_rows = len(matrix)

# Get the number of columns (assuming all rows have the same length)
num_cols = len(matrix[0])

# Print the shape
print((num_rows, num_cols))  


# To slice a matrix in Python, you can use nested loops to 
# iterate over the rows and columns of the matrix and extract the desired elements.


the_matrix = [[1, 2, 3], [4, 5, 6], [6, 7, 8]]
the_matrix[0][0]


#Another example

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Slice the matrix to get the 2x2 sub-matrix in the bottom-right corner
sub_matrix = []
for i in range(1, 3):
    row = []
    for j in range(1, 3):
        row.append(matrix[i][j])
    sub_matrix.append(row)

# Print the sub-matrix
print(sub_matrix) 


#To find the axis of a matrix in Python, you can use the same approach 
# as for finding the shape, since the number of rows corresponds 
# to the first axis and the number of columns corresponds to the second axis.


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Get the number of rows (which corresponds to the first axis)
num_rows = len(matrix)

# Get the number of columns (which corresponds to the second axis)
num_cols = len(matrix[0])

# Print the axis of the matrix
print((num_rows, num_cols))  


#Some of the key operations that can be performed on matrices include addition, subtraction, scalar multiplication, and matrix multiplication.
#  Below a function that adds two matrices. 

# A function that adds two matrices 
def add_matrices(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None  # matrices have different shapes
    
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)
    
    return result

mat1 = [[1,2,3], [4,5,6], [7,8,9]]
mat2 = [[9,8,7], [6,5,4], [3,2,1]]

print(add_matrices(mat1, mat2))


# The function takes two matrices as input and returns their product if they are valid for multiplication,
#  and returns None otherwise. It first checks if the number of columns in the first matrix 
# matches the number of rows in the second matrix, which is a requirement for matrix multiplication. 
# Then, it initializes a result matrix with zeros, and computes the dot product of each row in the first matrix
#  with each column in the second matrix. Finally, it populates the result matrix with the computed dot products and returns it.



def matrix_multiplication(mat1, mat2):
    """
    Perform matrix multiplication of two matrices 
    """
    m1_rows, m1_cols = len(mat1), len(mat1[0])
    m2_rows, m2_cols = len(mat2), len(mat2[0])

    # Check if matrices are valid for multiplication
    if m1_cols != m2_rows:
        return None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(m2_cols)] for _ in range(m1_rows)]

    # Compute the dot product of each row in the first matrix with each column in the second matrix
    for i in range(m1_rows):
        for j in range(m2_cols):
            dot_product = 0
            for k in range(m1_cols):
                dot_product += mat1[i][k] * mat2[k][j]
            result[i][j] = dot_product

    return result

mat1 = [[1,2,3], [4,5,6], [7,8,9]]
mat2 = [[9,8,7], [6,5,4], [3,2,1]]
print(matrix_multiplication(mat1, mat2))


#NUMPY
import numpy as np

# create a vector using NumPy
vector = np.array([1, 2, 3])
print(vector)

# create two vectors using NumPy
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# add the two vectors
result = v1 + v2
print(result)

# subtract the two vectors
result = v1 - v2
print(result)

# multiply a vector by a scalar
result = 2 * v1
print(result)





# create a matrix using NumPy
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)



# create two matrices using NumPy
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# add the two matrices
result = m1 + m2
print("Add matrices\n", result, "\n")

# subtract the two matrices
result = m1 - m2
print("Subtract matrices\n", result, "\n")

# multiply a matrix by a scalar
result = 2 * m1
print("Multiply matrix by scalar \n", result, "\n")

# multiply two matrices
result = np.dot(m1, m2)
print("Multiply matrices\n", result)



# get the shape of the matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
shape = matrix.shape
print(shape)




v = np.array([1, 2, 3, 4, 5])

# Slice the first three elements of the vector
v_first_three = v[:3]
print(v_first_three)

# Slice the last three elements of the vector
v_last_three = v[-3:]
print(v_last_three)

# Slice middle three elements 
v_middle_three = v[1:4]
print(v_middle_three)




m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slice the first row of the matrix
m_sliced = m[0, :]

print(m_sliced)  # [1 2 3]

# Slice the second column of the matrix
m_sliced = m[:, 1]

print(m_sliced)  # [2 5 8]

# Slice a 2x2 submatrix of the matrix
m_sliced = m[1:, :2]

print(m_sliced)  # [[4 5], [7 8]]
