matrix = [[1,2,3], [4,5,6], [7,8,9]]

for row in matrix: #zakonisht keshtu iterojme ne matrixes
    for el in row:
        print(el, end = "")
    print()



# python3 emri.py - ekzekuton kodin
#chmod u+x - userit i jep te drejten e ...?
#./emri.py - ekzekuton si script
#testimi automatik i kodit ne git    


#Task 1 - iterate on each row me rradhe, append colums 3,4 to middle
# cat emri.py - shfaq permbajtjen e file-it
#chmod u+x *.py - per gjithe filet, pasi e krijon filein ||  e ben file-in qe e ka chmod ne rregull dhe e kopjon, fshin permbajtjen dhe e perdor si file te ri - cp name.py newfile.py
#te Task 2 matrix eshte 3 dimensionale, prap lista 1 , e verdha ka x roze, roza ka x blu dhe blu ka x elemente || sa kllapa ke
#numri i rreshtave i matrix eshte len(), cols = len(matrix[0])
    rows = len(matrix)
    cols = len(matrix[0])

    return (rows, cols)


    #duhet ti zvogelojme dimensionin?

    #me tab ploteson ne Linux


  #  i , kur nuk ke nje variabel e shkruan _


#Task 3
    #create empty matrix with same correct dims 
    #iterate over rows
    #iterate over cols
    #transpose [j][i] = matrix [i][j]

    mat1 =  matrix[1:3, :] #fillimisht rreshtat, pastaj kolonat


my_list = [1, 2, 3, 4]
my_tuple = (1,2)

#axis = 0 dmth sipas rreshtavee, axis =1 dmth sipas kolonave. kur nuk jep nje axis dmth = 0, kerko dimensonet pse duhen same shape

ne python rastet if etj etj te gabimit behen ne fillim qe kodi te mos vazhdoje



def def cat_matrices2D(mat1, mat2, axis=0): t
    """
A function that concatenates two matrices along a specific axis
    """

    if axis == 0:
        if len(mat1[0] != len(mat2[0])):
            return None

            

    for i in range(mat1_rows): #ecim te rreshtat me i


            dot_product = sum(mat1[i][k] * mat2[k][j] for k in range (mat1_columns)) #ate qe kane te perbashket