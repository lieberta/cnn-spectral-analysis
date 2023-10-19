import numpy as np
import matplotlib.pyplot as plt
import os
import csv

#initialize lists for each model: [trainepochs,trainloss,valepochs,valloss]
CNN3D, CNN1D3D, CNN3D1D = [[],[],[],[]], [[],[],[],[]], [[],[],[],[]]

# load data CNN1D3D into arrays
dataCNN1D3D = np.load('./Plots/numpysaves/CNN1D3D.npz')
CNN1D3D[0] = dataCNN1D3D['trainx']+1
CNN1D3D[1] = dataCNN1D3D['trainy']
CNN1D3D[2] = dataCNN1D3D['valx']+1
CNN1D3D[3] = dataCNN1D3D['valy']

# load data CNN3D1D into arrays
dataCNN3D1D = np.load('./Plots/numpysaves/CNN3D1D.npz')
CNN3D1D[0] = dataCNN3D1D['trainx']+1
CNN3D1D[1] = dataCNN3D1D['trainy']
CNN3D1D[2] = dataCNN3D1D['valx']+1
CNN3D1D[3] = dataCNN3D1D['valy']

# load data CNN3D into arrays
dataCNN3D = np.load('./Plots/numpysaves/CNN3D.npz')
CNN3D[0] = dataCNN3D['trainx']+1
CNN3D[1] = dataCNN3D['trainy']
CNN3D[2] = dataCNN3D['valx']+1
CNN3D[3] = dataCNN3D['valy']

# load data CNN3D1D into arrays
dataCNN3D1D = np.load('./Plots/numpysaves/CNN3D1D.npz')
CNN3D1D[0] = dataCNN3D1D['trainx']+1
CNN3D1D[1] = dataCNN3D1D['trainy']
CNN3D1D[2] = dataCNN3D1D['valx']+1
CNN3D1D[3] = dataCNN3D1D['valy']


#takes a Model List (CNN3D1D or CNN1D3D) and gives out the improvement in comparison to CNN3D
def calculate_improvementlist(List):
    list=[]
    for i, _ in enumerate(List[2]):
        improvement = round((List[3][i] - CNN3D[3][i]) / List[3][i] * 100,2) # the improvement in epoch i+1 rounded to 2 decimals
        list.append(improvement)
    return list


def create_plot():
    plt.plot(CNN1D3D[0], CNN1D3D[1], color='green', linestyle= '-', label='Trainloss CNN1D3D')
    plt.plot(CNN1D3D[2], CNN1D3D[3], color='green', linestyle= '--', label='Validationloss CNN1D3D')

    plt.plot(CNN3D1D[0], CNN3D1D[1], color='purple', linestyle= '-', label='Trainloss CNN3D1D')
    plt.plot(CNN3D1D[2], CNN3D1D[3], color='purple', linestyle= '--', label='Validationloss CNN3D1D')

    plt.plot(CNN3D[0], CNN3D[1], color='red', linestyle= '-', label='Trainloss CNN3D')
    plt.plot(CNN3D[2], CNN3D[3], color='red', linestyle= '--', label='Validationloss CNN3D')

    plt.legend()
    plt.title('Loss per Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')


    plotfilename = f'./Plots/Evaluation/Vergleich.png'
    i = 0 # filename additional number
    while os.path.exists(plotfilename):
        # if the file exists, add a number to the file name
        i+=1
        plotfilename= f'./Plots/Evaluation/Vergleich_version_{i}.png'

    plt.savefig(plotfilename)
    plt.show()


# creates a .csv with every i-th entry, multiply loss with e3 and round it to 4 decimals
def create_csv(modulo):
    improvementsCNN1D3D = calculate_improvementlist(CNN1D3D)
    improvementsCNN3D1D = calculate_improvementlist(CNN3D1D)

    new_CNN1D3D, new_CNN3D, new_CNN3D1D = [[],[],[],[]],[[],[],[],[]],[[],[],[],[]]
    #create empty lists for val_epoch
    new_CNN1D3D[2], new_CNN3D[3], new_CNN1D3D[3],new_CNN3D1D[3], new_improvCNN1D3D, new_improvCNN3D1D = [],[],[],[],[],[]

    for i, _ in enumerate(CNN1D3D[2]):
        if i%modulo ==modulo-1:
            new_CNN1D3D[2].append(CNN1D3D[2][i])
            #new_CNN3D1D[2].append(CNN3D1D[2][i])

            new_CNN3D[3].append(round(CNN3D[3][i]*1000,4))
            new_CNN1D3D[3].append(round(CNN1D3D[3][i]*1000,4))
            new_CNN3D1D[3].append(round(CNN3D1D[3][i] * 1000, 4))

            new_improvCNN1D3D.append(improvementsCNN1D3D[i])
            new_improvCNN3D1D.append(improvementsCNN3D1D[i])


    csv_filename = './Plots/Evaluation/vergleich.csv'

    # Zip the lists together and insert the column names as the first row
    data = [['Epochs', 'Validationloss CNN3D (e-3)', 'Validationloss CNN1D3D/proposed method (e-3)','Validationloss CNN3D1D (e-3)','Improvement CNN1D3D (%)','Improvement CNN3D1D (%)']] + list(
        zip(new_CNN1D3D[2], new_CNN3D[3], new_CNN1D3D[3], new_CNN3D1D[3],new_improvCNN1D3D,new_improvCNN3D1D))

    i = 0 # filename additional number
    while os.path.exists(csv_filename):
        # if the file exists, add a number to the file name
        i+=1
        csv_filename = f'./Plots/Evaluation/vergleich{i}.csv'

    # Open the file for writing
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
    create_plot()
    #create_csv(5)
