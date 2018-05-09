############################################### 
# 학과:  컴퓨터공학과
# 학번:  20143085
# 이름:  이대현
###############################################  
# forward elimination
 # 1. 초기 pivot의 위치는 [0, 0]으로 한다.
  # 2. pivot의 위치를 재설정 한다. 
  #         주어진 augmented matrix M에 대하여 행과 열의 수는 m과 n+1이라고 한다.
  #         2-1. 현재 pivot의 위치가 [i, j] (단, i < m, j < n)에 있다고 가정하자. 
  #  2-2. 만일 [i, j]의 값이 0이 아닌 값이면, 현재 pivot의 위치는 재설정할 필요가 없으므로, 3의 과정으로 간다. 
  #  2-3. 만일 [i, j]의 값이 0이면, pivot의 아래 행을 방문하며 [p, j] (단, p > i) 값을 검사한다. 
  #              만일 [p, j]의 값이 0이 아닌 값이면 i행과 p행을 서로 교체(interchange)한다. 
  #  2-4. 만일 2-3 과정에서 [p, j]의 값이 0이 아닌 값을 못 찾았을 경우, pivot의 위치를 한 열 옆으로  
  #              이동(j <- j+1) 한다. 
  # 2-5. 0값이 아닌 pivot의 위치를 찾을 때까지 2-1 ~ 2-4의 과정을 반복한다. 
  #      만일, pivot의 위치가 행렬의 마지막 행과 끝에서 두 번째 열인 경우에는 주어진 행렬의 마지막 pivot이 되며, 
  #      이후 다음 pivot은 설정하지 않고 forward elimination을 종료한다.  
  # 3. pivot의 값이 a라면 그 행의 모든 값에 1/a를 곱하여 scaling을 수행한다. 
  # 4. pivot의 위치보다 아래 행들의 같은 열의 값이 0이 되도록 replacement를 수행한다. 
  # 5. replacement 과정이 모두 끝나면 pivot의 위치를 다시 재설정 한다.
  #   기존의 pivot위치에서 행은 한 행 아래로 내려가고, 열은 아래 행들 중에 왼쪽으로부터 모든 열의 값에서 하나라도 0이 아닌 곳으로 가도록 설정한다. 
  # 6. 모든 행렬이 행 사다리꼴(row echelon form)이 될 때까지 과정[2 ~ 5]를 반복한다. 


def forward_elimination(matrix):  
  ###############################################  
  # 아래 라인을 지우고 forward elimination을 구현하세요.  
  ############################################### 
  j=0  
  m,n = matrix.shape
  for i in range(0, m):
      for p in range(j,n-1):
         if(is_nearzero(matrix[i,p], tol = 1e-05)):
            for l in range(i+1, m):
              if(not is_nearzero(matrix[l,p], tol = 1e-05)):
                interchange(matrix, i, l)
                scaling(matrix, i, 1/float(matrix[i,p]))

                for k in range(i+1, m):
                    replacement(matrix, k, i, matrix[k, p])
                break
        
         else:      
            scaling(matrix, i, 1/float(matrix[i,p]))

            for k in range(i+1, m):
                    replacement(matrix, k, i, matrix[k, p])
         if(matrix[i,p]==1):
          break      
        

    

  return matrix
  
  

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

# forward elimination이 적용된 사다리꼴 행렬(row echelon form)을 이용하여 system matrix A와 augmented matrix M의 rank를 구한다. 
def rank(aug_mat):
    m, n = aug_mat.shape  
    rank_A = 0  
    rank_M = 0  
    sys_mat = aug_mat[:, 0:n-1]  

    #System matrix A의 rank 측정.  
    for r in range(0, m):   
      if (not is_zerorow(sys_mat, r)):    
        rank_A +=1  

    #Augmented matrix M의 rank 측정.   
    for r in range(0, m):   
    	if (not is_zerorow(aug_mat, r)):    
 		   rank_M +=1  
    return rank_A, rank_M    

############################################### 
# 프로그램 메인 파트 
############################################### 
import numpy, sys  

input_file = sys.argv[1]
output_file = sys.argv[2]

aug_matrix = numpy.mat(numpy.loadtxt(input_file))  
# 입력 augmented matrix (aug_matrix)는  
# 아래의 forward_elimination() 함수를 수행한 후, 
# row echelon form으로 바뀜. 
forward_elimination(aug_matrix)  
m, n = aug_matrix.shape 

n-=1
rank_A, rank_M = rank(aug_matrix) 
############################################### 
# rank_A와 rank_M을 이용해서 주어진 선형 시스템이  
# (‘Inconsistent’, ‘A unique solution’, ‘Infinity many solutions’) 중 
# 어느 경우에 해당하는지를 판단한다.  
# 
# Augmented Matrix(M) = [A|b] 
# rank_A < rank_M then, Linear system is inconsistent 
# rank_A = rank_M = n, Linear system has a unique solution 
# rank_A = rank_M < n, Linear system has infinity many solutions 
# 
# 아래 라인을 지우고 위의 내용을 구현하세요. 
############################################### 
if (rank_A < rank_M):
  FE_result = "Inconsistent"
elif (rank_A == rank_M and rank_A == n):
  FE_result = "A unique solution"
else:
  FE_result = "Infinity many solution"

############################################### 
# 출력파일의 형식을 맞추기 위해서 다음 코드는 절대로 건들지 마세요.
 ############################################### 
f = open(output_file, 'a') 
f.write("Rank 계산 결과\n") 
f.write("rankA = ") 
f.write(str(rank_A)) 
f.write("\nrankM = ") 
f.write(str(rank_M)) 
f.write("\nThe number of n is ") 
f.write(str(n)) 
f.write("\nForward_elimination 결과 = ")
f.write(str(FE_result)) 
# FE_result는 문자열로써, ‘Inconsistent’, ‘A unique solution’, ‘Infinity many solutions’ 중 하나임. 
f.write("\nRow echelon form matrix is = \n") 
f.write(str(aug_matrix)) 
f.close()
############################################### 