import torch
from dataset import Dataset_x4_y1, Dataset_x1_y1
from models import CNN1D3D, CNN3D, CNN3D1D
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import math

from import_normalise import npz_to_matrixlist


min = 19.998600006103516
dist = 1344.7513999938965

experiment = 3 # number of experiment
n_timesteps = 60 # number of timesteps
input_timesteps = 4 # number of input timesteps for CNN1D3D

#reverse the normalization process
def denormalize(tensor):
    tensor_denorm = tensor * dist +min
    return tensor_denorm




device = ("cuda" if torch.cuda.is_available() else "cpu")
dataset1 = Dataset_x4_y1()
dataset2 = Dataset_x1_y1()


#load model and dataloader to the gpu
modelCNN1D3D = CNN1D3D().to(device)
modelCNN3D = CNN3D().to(device)
modelCNN3D1D = CNN3D1D().to(device)

modelCNN1D3D, modelCNN3D, modelCNN3D1D = modelCNN1D3D.double(), modelCNN3D.double(), modelCNN3D1D.double() #for no 'expected double but got float' error

modelCNN1D3D.load_state_dict(torch.load('./saved_models/savedCNN1D3D'))
modelCNN3D.load_state_dict(torch.load('./saved_models/savedCNN3D'))
modelCNN3D1D.load_state_dict(torch.load('./saved_models/savedCNN3D1D'))

data_loader1 = DataLoader(dataset=dataset1, shuffle=False, batch_size=1)
data_loader2 = DataLoader(dataset=dataset2, shuffle=False, batch_size=1)


def plot_pointcloud(t):
    with torch.no_grad():

        for i, (input,target) in enumerate(data_loader):
            if i ==t:
                input = input.to(device)
                target = target.to(device)
                output = model1(input.double())

                input = denormalize(input)
                target = denormalize(target)
                output = denormalize(output)

                error = abs(target-output)

                #print(f'min: {torch.min(error)}, max: {torch.max(error)}')
                e_min, e_max = torch.min(error), torch.max(error)

                #from 1,1,121,161,61 to 121,161,61 tensor
                error = error.reshape(121,161,61)

                #scale error with 2 times the mean value of the error
                #every error above two times the meanerror is scaled to 1
                m_error= torch.mean(error)
                error = (error- e_min)/(2*m_error)
                error[error >1] = 1

                error = error.numpy()


                # just look at every 10th element so new size: (13,17,7)
                error = error[::10,::10,::10]


                x,y,z = error.shape


                e_list = error.flatten()
                zero_list = np.full((x,y,z),0).flatten()
                e_list = list(zip(e_list,1-e_list,zero_list))



                #lets plot:
                fig = plt.figure()

                ax = fig.add_subplot(111, projection='3d')

                x, y, z = np.meshgrid(range(x), range(y), range(z))


                ax.scatter3D(x, y, z, c=e_list, s=10) #s= scaling of points
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                a1,a2 = 45,215
                ax.view_init(a1,a2)
                #for angle in range(0, 360):
                #    ax.view_init(30, angle)
                #    plt.draw()
                #    plt.pause(.001)

                #plt.savefig(f'./Plots/pointclouds/meanerror={m_error}_t={i}.png')
                plt.show()


                break


# durchschnittliche abweichung
def average_devation():
    with torch.no_grad():

        deviation_CNN3D = []#np.array([])
        deviation_CNN1D3D= []#np.array([])
        deviation_CNN3D1D= []#np.array([])


        j=0
        for i, (input, target) in enumerate(data_loader1):
            # the amount of (input,target) packages in a single experiment is here n_timesteps-input_timesteps
            # therefore for each experiment, take the (input, target) pairs out of the right thing
            if i in range(experiment * (n_timesteps-input_timesteps), (experiment + 1) * (n_timesteps-input_timesteps)):
                j+=1
                input = input.to(device)
                input= input.double()
                target = target.to(device)
                target= target.double()

                #print(f"inputsize= {input.size()}")
                outputCNN3D1D = modelCNN3D1D(input)
                outputCNN1D3D = modelCNN1D3D(input)
                outputCNN1D3D = denormalize(outputCNN1D3D)
                outputCNN3D1D = denormalize(outputCNN3D1D)
                target = denormalize(target)

                m1 = torch.mean(abs(outputCNN1D3D - target))
                m1 = m1.cpu().numpy()
                print(f'm1={m1}')
                m2 = torch.mean(abs(outputCNN3D1D - target))
                m2 = m2.cpu().numpy()

                deviation_CNN1D3D.append(m1)
                deviation_CNN3D1D.append(m2)
                #np.append(deviation_CNN1D3D, m1)
                #np.append(deviation_CNN3D1D, m2)
                if j == n_timesteps -4:
                    break
        deviation_CNN1D3D, deviation_CNN3D1D = np.array(deviation_CNN1D3D), np.array(deviation_CNN3D1D)


        j = 0
        for i, (input, target) in enumerate(data_loader2):
            if i in range(experiment * (n_timesteps - 1), (experiment + 1) * (n_timesteps - 1)):
                j += 1
                input = input.to(device)
                target = target.to(device)
                target = target.double()
                output = modelCNN3D(input.double())

                target = denormalize(target)
                output = denormalize(output)

                m3 = torch.mean(abs(output - target))
                m3 = m3.cpu().numpy()

                deviation_CNN3D.append(m3)



                if j == 59:
                    break
        deviation_CNN3D = np.array(deviation_CNN3D)


        return np.mean(deviation_CNN1D3D), np.mean(deviation_CNN3D1D), np.mean(deviation_CNN3D)


def deviation_inpoint(x,y,z):
    with torch.no_grad():


        output_tempCNN3D1D = []
        target_temp = []
        diff_list = np.array([])




        j=0
        for i, (input, target) in enumerate(data_loader1):
            # the amount of (input,target) packages in a single experiment is here n_timesteps-input_timesteps
            # therefore for each experiment, take the (input, target) pairs out of the right thing
            if i in range(experiment * (n_timesteps-input_timesteps), (experiment + 1) * (n_timesteps-input_timesteps)):
                j+=1

                input = input.to(device)
                input= input.double()

                target =target.to(device)
                target = target.double()



                #print(f"inputsize= {input.size()}")


                outputCNN3D1D = modelCNN3D1D(input)

                outputCNN3D1D = denormalize(outputCNN3D1D)

                target= denormalize(target)



                output_tempCNN3D1D.append(outputCNN3D1D[0, 0, x, y, z].cpu().numpy())
                target_temp.append(target[0, 0, x, y, z].cpu().numpy())


                if j == n_timesteps -4:
                    break

        diff_list = abs(np.array(output_tempCNN3D1D)-np.array(target_temp))

        average =  np.mean(diff_list)

        return average







        ax = plt.subplot(1, 1, 1)

        # print(f'realtemp1 length: {len(real_temp1)} \n realtemp2 length: {len(real_temp2)}')



        # plot for CNN3D
        ax.plot(range(2,61),real_temp, '-bo', label='real temp',markersize =2)
        # target data
        ax.plot(range(2,61), output_tempCNN3D, '-ro', label= 'temp forecast CNN3D',markersize=2)
        # plot for CNN1D3D
        ax.plot(range(5, 61), output_tempCNN1D3D, '-go', label='temp forecast CNN1D3D', markersize=2)

        ax.plot(range(5, 61), output_tempCNN3D1D, '-mo', label='temp forecast CNN3D1D', markersize=2)




# plot target temperature and actual temperature of one single room point (x,y,z) over time
def plot_tempgraph(x,y,z):
    with torch.no_grad():


        output_tempCNN1D3D = []
        output_tempCNN3D1D = []

        real_temp = []
        output_tempCNN3D = []


        dataiter = iter(data_loader2)
        sample_batch = dataiter.next()
        print(f'targetshape of dataloader2: {sample_batch[1].shape} \n length: {len(data_loader2.dataset)}')

        j=0
        for i, (input, target) in enumerate(data_loader1):
            # the amount of (input,target) packages in a single experiment is here n_timesteps-input_timesteps
            # therefore for each experiment, take the (input, target) pairs out of the right thing
            if i in range(experiment * (n_timesteps-input_timesteps), (experiment + 1) * (n_timesteps-input_timesteps)):
                j+=1

                input = input.to(device)
                input= input.double()



                #print(f"inputsize= {input.size()}")


                outputCNN3D1D = modelCNN3D1D(input)
                outputCNN1D3D = modelCNN1D3D(input)


                outputCNN1D3D = denormalize(outputCNN1D3D)
                outputCNN3D1D = denormalize(outputCNN3D1D)
                output_tempCNN1D3D.append(outputCNN1D3D[0, 0, x, y, z].cpu().numpy())
                output_tempCNN3D1D.append(outputCNN3D1D[0, 0, x, y, z].cpu().numpy())
                if j == n_timesteps -4:
                    break

        dataiter = iter(data_loader1)
        sample_batch = dataiter.next()
        print(f'targetshape of dataloader1: {sample_batch[1].shape} \n length: {len(data_loader1.dataset)}')


        j=0
        for i, (input, target) in enumerate(data_loader2):
            if i in range(experiment*(n_timesteps-1),(experiment+1)*(n_timesteps-1)):
                j+=1
                input = input.to(device)
                target = target.to(device)
                output = modelCNN3D(input.double())

                target = denormalize(target)
                output = denormalize(output)



                real_temp.append(target[0, 0, x, y, z].cpu().numpy())
                output_tempCNN3D.append(output[0, 0, x, y, z].cpu().numpy())

                if j == 59:

                    break




        ax = plt.subplot(1, 1, 1)

        # print(f'realtemp1 length: {len(real_temp1)} \n realtemp2 length: {len(real_temp2)}')



        # plot for CNN3D
        ax.plot(range(2,61),real_temp, '-bo', label='real temp',markersize =2)
        # target data
        ax.plot(range(2,61), output_tempCNN3D, '-ro', label= 'temp forecast CNN3D',markersize=2)
        # plot for CNN1D3D
        ax.plot(range(5, 61), output_tempCNN1D3D, '-go', label='temp forecast CNN1D3D', markersize=2)

        ax.plot(range(5, 61), output_tempCNN3D1D, '-mo', label='temp forecast CNN3D1D', markersize=2)



        plt.title(f'Point ({x},{y},{z})')
        plt.legend(loc="upper left")
        plt.xlabel('time')
        plt.ylabel('temperature')
        plt.savefig(f'./Plots/temp_graphs/Point ({x},{y},{z})targetCNN1D3D_experiment{experiment}.png')
        plt.show()



#for i in range (31):
    # wierd numbers for the ratio to get the true diagonal
    #plot_tempgraph(math.ceil(1.967*i),math.ceil(2.6129*i),i)

#plot_pointcloud(1)

#a = deviation_inpoint(50,66,25)
#print(a)

c1,c2,c3 = average_devation()
print(f'CNN1D3D = {c1}, CNN3D1D = {c2}, CNN3D = {c3}')


