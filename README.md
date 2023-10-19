# Geospatial Time Series

Here we try to forecast temperature development in a simulated room via CNNs. Data used to train the model are simulated 
temperature grid meshes of burning rooms.

Preprocessing of new Experiment folders:

use "out_to_csv.py" to translate the *.out files of one experiment to
a *.csv-file

use then  "sort_data.py" sort these gridpoints in (x,y,z) directions
and to save a "temperature_matrices.npz" file for each experiment

Train Network:

Execute "train.py" to start the training of given method, the model will be
saved in the folder "saved_models" afterwards 




Explanation of files:

"import_normalize.py" are predefined functions, that import the *.npz files
and normalize the matrices in it.
import_all_rooms(abl = boolean) imports all rooms and is executed in "dataset.py"
if abl == True, the dataset will consist of derivation of the temperature
if abl == False, the dataset will consist of the actual temperature values


