import numpy as np
from numpy import linalg as LA
from PIL import Image
from PIL import ImageFilter
import time

start_time = time.time()

print("\nImage Denoising Start...")

##### Parameter Setting ################################################################################################

new_size_H = 96
new_size_W = 96
image_name = 'ChunSewhan.jpg'
neighbor = "8"
theta = 1
r = 1
L_method = 0

########################################################################################################################





##### Function Definition ##############################################################################################

def Right(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i + 1] - new_image_np_flattened_normed[0,i])**2)/(2*(theta**2)))

def Left(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i - 1] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

def Upper(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i - new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

def Lower(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i + new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

def RightLower(i):  #prob?
    return np.exp(-1*((new_image_np_flattened_normed[0,i + 1 + new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2)/(2*theta**2))

def RightUpper(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i + 1- new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

def LeftLower(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i -1 + new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

def LeftUpper(i):
    return np.exp(-1 * ((new_image_np_flattened_normed[0,i -1 - new_image_np.shape[1]] - new_image_np_flattened_normed[0,i])**2) / (2*(theta**2)))

########################################################################################################################





##### Image Initialization #############################################################################################

# Image Load + Edit Image Size

print("    Image Initialization Start...")

ori_image = Image.open('./'+image_name).convert('L')
ori_image_np = np.array(ori_image)
ori_W,ori_H = ori_image.size

new_index_H = int(ori_H/new_size_H)
new_index_W = int(ori_W/new_size_W)
new_image = Image.new('L', (new_size_H,new_size_W))
new_image_pa = new_image.load()

for i in range(0, new_size_H):
    for j in range(0, new_size_W):
        new_image_pa[j,i] = int(np.mean(ori_image_np[new_index_H*i:new_index_H*i+new_index_H , new_index_W*j:new_index_W*j+new_index_W]))

new_image_np = np.array(new_image)

print("    Image Initialization Done!")

########################################################################################################################





##### Laplacian Matrix Calculation #####################################################################################

print("    Laplacian Matrix Calculation Start...")

Num_Pixel = new_size_H*new_size_W
new_image_np_flattened_normed=np.reshape(new_image_np,[1,Num_Pixel])/255
f = new_image_np_flattened_normed.T

# W Calculation

W=np.zeros([Num_Pixel,Num_Pixel])


if neighbor == "2_V":
    for i in range(0, Num_Pixel):

        # corner
        if i == 0:
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif i == new_image_np.shape[1] - 1:
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif i == Num_Pixel - new_image_np.shape[1]:
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif i == Num_Pixel - 1:
            W[i, i - new_image_np.shape[1]] = Upper(i)

        # Edge
        elif (i > 0) and i < (new_image_np.shape[1] - 1):  # Upper
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif (i > Num_Pixel - new_image_np.shape[1]) and (i < Num_Pixel - 1):  # Lower
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif ((i + 1) % new_image_np.shape[1]) == 0:  # Right
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif ((i + 1) % new_image_np.shape[1]) == 1:  # Left
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)

            # Mid-Range
        else:
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)

if neighbor == "2_H":
    for i in range(0, Num_Pixel):

        # corner
        if i == 0:
            W[i, i + 1] = Right(i)
        elif i == new_image_np.shape[1] - 1:
            W[i, i - 1] = Left(i)
        elif i == Num_Pixel - new_image_np.shape[1]:
            W[i, i + 1] = Right(i)
        elif i == Num_Pixel - 1:
            W[i, i - 1] = Left(i)

        # Edge
        elif (i > 0) and i < (new_image_np.shape[1] - 1):  # Upper
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
        elif (i > Num_Pixel - new_image_np.shape[1]) and (i < Num_Pixel - 1):  # Lower
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
        elif ((i + 1) % new_image_np.shape[1]) == 0:  # Right
            W[i, i - 1] = Left(i)
        elif ((i + 1) % new_image_np.shape[1]) == 1:  # Left
            W[i, i + 1] = Right(i)

            # Mid-Range
        else:
            W[i, i + 1] = Right(i)
            W[i, i - 1] = Left(i)

if neighbor == 4:
    for i in range(0, Num_Pixel):

        # corner
        if i == 0:
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif i == new_image_np.shape[1] - 1:
            W[i, i - 1] = Left(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif i == Num_Pixel - new_image_np.shape[1]:
            W[i, i + 1] = Right(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif i == Num_Pixel - 1:
            W[i, i - 1] = Left(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)

        # Edge
        elif (i > 0) and i < (new_image_np.shape[1] - 1):  # Upper
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
        elif (i > Num_Pixel - new_image_np.shape[1]) and (i < Num_Pixel - 1):  # Lower
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif ((i + 1) % new_image_np.shape[1]) == 0:  # Right
            W[i, i - 1] = Left(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
        elif ((i + 1) % new_image_np.shape[1]) == 1:  # Left
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)

            # Mid-Range
        else:
            W[i, i + 1] = Right(i)
            W[i, i - 1] = Left(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)

if neighbor == 8:
    for i in range(0, Num_Pixel):

        #corner
        if i == 0:
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i + new_image_np.shape[1]+1] = RightLower(i)
        elif i == new_image_np.shape[1]-1:
            W[i, i - 1] = Left(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i + new_image_np.shape[1]-1] = LeftLower(i)
        elif i ==Num_Pixel-new_image_np.shape[1]:
            W[i, i + 1] = Right(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1]+1] = RightUpper(i)
        elif i ==Num_Pixel-1:
            W[i, i - 1] = Left(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1]-1] = LeftUpper(i)

        #Edge
        elif (i > 0) and i < (new_image_np.shape[1]-1): #Upper
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i + new_image_np.shape[1] - 1] = LeftLower(i)
            W[i, i + new_image_np.shape[1] + 1] = RightLower(i)
        elif (i > Num_Pixel-new_image_np.shape[1]) and (i < Num_Pixel-1): #Lower
            W[i, i - 1] = Left(i)
            W[i, i + 1] = Right(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1] - 1] = LeftUpper(i)
            W[i, i - new_image_np.shape[1] + 1] = RightUpper(i)
        elif ((i+1) % new_image_np.shape[1]) == 0: #Right
            W[i, i - 1] = Left(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1] - 1] = LeftUpper(i)
            W[i, i + new_image_np.shape[1] - 1] = LeftLower(i)
        elif ((i+1) %new_image_np.shape[1]) == 1: #Left
            W[i, i + 1] = Right(i)
            W[i, i + new_image_np.shape[1]] = Lower(i)
            W[i, i - new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1] + 1] = RightUpper(i)
            W[i, i + new_image_np.shape[1] + 1] = RightLower(i)

            #Mid-Range
        else:
            W[i, i+1] = Right(i)
            W[i, i-1] = Left(i)
            W[i, i+new_image_np.shape[1]] = Lower(i)
            W[i, i-new_image_np.shape[1]] = Upper(i)
            W[i, i - new_image_np.shape[1] + 1] = RightUpper(i)
            W[i, i + new_image_np.shape[1] + 1] = RightLower(i)
            W[i, i - new_image_np.shape[1] - 1] = LeftUpper(i)
            W[i, i + new_image_np.shape[1] - 1] = LeftLower(i)

D = np.zeros_like(W)
for i in range(0, Num_Pixel):
    D[i, i] = np.sum(W[i, :])
L = D - W

print("    Laplacian Matrix Calculation Done!")


########################################################################################################################





##### Eigen Value Calculation ##########################################################################################

print("    Eigen Value Calculation Start...")

eigvals, eigvecs = np.linalg.eig(L)
eigvals = np.round(eigvals,7).real
eigvecs = np.round(eigvecs,7)

print("    Eigen Value Calculation Done!")

########################################################################################################################





##### h Value Calculation ##############################################################################################

print("    h_lamda / h_L Value Calculation Start...")

h_lamda = np.zeros([eigvals.shape[0],eigvals.shape[0]])
for i in range (0,eigvals.shape[0]):
    h_lamda[i,i] = 1/((r*(eigvals[i]))+1)
h_lamda = np.round(h_lamda,7)

t = np.matmul(eigvecs,h_lamda)
h_L = np.matmul(t,eigvecs.transpose())

print("    h_lamda / h_L Value Calculation Done!")

########################################################################################################################





##### Answer Calculation ###############################################################################################

print("    Final Calculation Start...")

Answer = np.matmul(h_L,f*255)
Answer = np.reshape(Answer,[new_size_H,new_size_W]).astype(np.uint8)
Answer = Image.fromarray(Answer, 'L')
#Answer.show()
Answer.save("./"+ "Result_Image_"+"theta_" +str(theta) +"_r_"+ str(r) +"_"+str(neighbor)+"N_"+str(new_size_W)+ ".png")

print("    Final Calculation Done!")

########################################################################################################################


print("Image Denoising Done!\n")

print("--- %s seconds ---" % (round(time.time() - start_time,1)))