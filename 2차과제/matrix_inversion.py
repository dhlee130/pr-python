###############################################
# 학과:  컴퓨터 공학과
# 학번: 20143085  
# 이름: 이대현
###############################################  
# 주어진 n×n 정사각행렬 𝐀로부터 augmented matrix 𝐌=[𝐀|𝐈]를 생성한다. 
# 생성된 augmented matrix 𝐌에 대해 forward elimination을 수행하여 행 사다리꼴 행렬(row echelon form)을 계산한다. 
# 행 사다리꼴 행렬에 대해, rank 𝐀를 계산한다. 
# 계산된 rank 𝐀와 n을 이용하여, 주어진 n×n 정사각행렬 𝐀가 역행렬을 가질 수 있는지(Invertible) 아니면 역행렬을 가질 수 없는지 (Singular) 여부를 판단한다. 
# 정사각행렬 𝐀가 역행렬을 가질 수 있다면, 현재 augmented matrix 𝐌에 대해 backward elimination을 수행한다. 
# backward elimination을 수행하여 얻은 [𝐈|𝐀−𝟏]에서 𝐀의 역행렬인 𝐀−𝟏을 구한다. 
def matrix_inversion(matrix):  
  m, n  = matrix.shape

  aug_mat = numpy.mat(adjoin_I_to_A(matrix))

  forward_elimination(aug_mat)



  rank_A = rank(aug_mat[0:n, 0:n])    
  if (rank_A == n):   
 	         result = "Invertible"  
 	         backward_elimination(aug_mat)   
 	         matrix = numpy.mat(readoff_inv_A(aug_mat))  

  elif(rank_A < n):
 	           result = "Singular"    
 	     
  return result, matrix  

# n×n square matrix 𝐀 의 오른쪽에 같은 크기의 identity matrix 𝐈 을 인접하여 [𝐀|𝐈] 형태로 만든다.  
# 인접한 [𝐀|𝐈] 행렬의 행의 수는 n, 열의 수는 2n. 

def adjoin_I_to_A(matrix):
  m, n = matrix.shape
  identity_matrix = numpy.mat(numpy.identity(n))




  aug_mat = numpy.mat(numpy.arange(n*n*2).reshape((n, n*2)) , dtype = float)
  aug_mat[0:n,0:n] = matrix
  aug_mat[0:n,n:n*2] = identity_matrix  


  return aug_mat  

# Gauss-Jordan 소거법이 적용된 [𝐈|𝐀−𝟏]에서 𝐀−𝟏을 구한다. 
def readoff_inv_A(aug_mat):
  	m, n = aug_mat.shape  


  	return aug_mat[0:m, m:m*2]  

# forward elimination 
def forward_elimination(aug_mat):  
  ######################################################### 
  # 아래 라인을 지우고 forward elimination을 구현하세요.  
  #########################################################  
  j=0  
  m,n = aug_mat.shape
  for i in range(0, m):
      for p in range(j,n-1):
         if(is_nearzero(aug_mat[i,p], tol = 1e-05)):
            for l in range(i+1, m):
              if(not is_nearzero(aug_mat[l,p], tol = 1e-05)):
                interchange(aug_mat, i, l)
                scaling(aug_mat, i, 1/float(aug_mat[i,p]))

                for k in range(i+1, m):
                    replacement(aug_mat, k, i, aug_mat[k, p])
                break
        
         else:      
            scaling(aug_mat, i, 1/float(aug_mat[i,p]))

            for k in range(i+1, m):
                    replacement(aug_mat, k, i, aug_mat[k, p])
         if(aug_mat[i,p]==1):
          break      

  return aug_mat

# backward elimination 
def backward_elimination(aug_mat): 
 #########################################################  
 # 아래 라인을 지우고 backward elimination을 구현하세요.  
 #########################################################  
  m,n = aug_mat.shape

  a=m-1
  b=int(n/2 - 1)
  c=0
  for i in range(0, a):
    for j in range(c, b):
      if(aug_mat[a-i, b-j] == 0):
        c+=1
        break
      elif(aug_mat[a-i, b-j] != 1):
        scaling(aug_mat, a-i, 1/float(aug_mat[a-i, b-j]))

      for k in range(1,a-i+1):
        replacement(aug_mat, (a-i)-k, a-i, aug_mat[(a-i)-k, b-j])
      c+=1
      break
  

  
  

  
 
  return aug_mat


# ri↔ rj 
def interchange(matrix, row_i, row_j):  
		m, n = matrix.shape  

		for i in range(0, n):  
		 	 matrix[row_i, i] ,matrix[row_j, i] = matrix[row_j, i] ,matrix[row_i, i]   

		return matrix  


# ri←cri 
def scaling(matrix, row_i, c): 
	 matrix[row_i, :] *= c  

	 return matrix  

# ri←ri−mrj 
def replacement(matrix, row_i, row_j, m): 
	 matrix[row_i, :] = matrix[row_i,:] - m*matrix[row_j, :]

	 return matrix  

# 행렬의 한 요소 값이 절대값 0.00001 보다 작으면 0과 가까운 숫자라고 판단한다. 
def is_nearzero(value, tol = 1e-05):  
  if(abs(value) < tol):   
    return True  
  else:   
    return False   

# i번째 행(i-th row)의 모든 요소가 0인지 검사한다. 
def is_zerorow(matrix, ith_row):  
	m, n = matrix.shape  

	for r in range(0, n):   
		if(not is_nearzero(matrix[ith_row, r])):   
			 return False  

	return True  

# 주어진 matrix에 대한 rank를 구하는 알고리즘 
def rank(matrix): 
  m, n = matrix.shape

  rank = 0  
 	
 	# 주어진 matrix에 대하여 rank 측정.  
 	

  for r in range(0, m):
      if (not is_zerorow(matrix, r)):
  		    rank +=1  

  return rank

############################################### 
# 프로그램 메인 파트 
############################################### 
import numpy, sys  

input_file = sys.argv[1] 
output_file = sys.argv[2]  

square_matrix = numpy.mat(numpy.loadtxt(input_file)) 

# 입력 square matrix 𝐀는 
# matrix_inversion()함수를 수행한 후, 
# result('Invertible', 'Singular' 중 하나)와  𝐀의 역행렬인 𝐀−𝟏를 반환한다. 

result, inverse_matrix = matrix_inversion(square_matrix) 

############################################### 
# 출력파일의 형식을 맞추기 위해서 다음 코드는 절대로 건들지 마세요. 
############################################### 

f = open(output_file, 'w') 
f.write(result) 
if(result=="Invertible"):  
 	f.write("\nInverse of the matrix A is = \n")  
 	f.write(str(inverse_matrix)) 
f.close() 
############################################### 