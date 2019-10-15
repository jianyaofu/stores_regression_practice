# stores_regression_practice

## Data Description
The data stores consists of 5 columns: tv, radio and newspaper are numerical variables, which stand for the advertising budgets for each media; region is a categorical variable and it value includes north, south, east and west; sales is a numerical column standing for the sale of each store. The dataset has 200 rows. We will use the first four variables to predict sales. The sample rows are shown below.
```
+---+-----+-------------+-------------+--------+-------------+
|   | tv  | radio       | newspaper   | region | sales       |
|---+-----+-------------+-------------+--------+-------------+
| 0 | 0.7 | 41.52659013	| 11.55986803 | north  | 10.57045796 |
| 1 | 4.1 | 13.78077247	| 4.198759861 | east   | 5.695451931 |
+---+-----+-------------+-------------+--------+-------------+
```

## Modules
### train_eval_split.py
Split the dataset into training and evaluating set based on the given fraction of train dataset size. It will create a directory and save the training and evaluating set as csv files.

Input: name of the original dataset in csv file, namely stores.csv and the fraction of train dataset between 0 and 1

Output: the ./data/ directory containing trian_data.csv and eval_data.csv

### trainer/constants.py
Define the input columns presented in the csv file as well as their default values and model target column.

### trainer/featurizer.py
Create the tensorflow featurized columns to be input into the linear regression model.

### trainer/input.py
Define data input functions to read data from csv and tfrecords files, parsing functions to convert csv and tf.example to tensors, and prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances.

### trainer/model.py
Design the simple linear regression model.

### trainer/task.py
Train the model on training set and evaluate it on evaluating set. Print the loss in each display step as well as the final stage.

## Usage
### Directory
```
 └── trainer
 │   ├── constants.py
 │   ├── featurizer.py
 │   ├── input.py
 │   ├── model.py
 │   └── task.py
 ├── data (created by running train_eval_split.py)
 │   ├── train_data.csv
 │   └── eval_data.csv
 ├── train_eval_split.py
 └── stores.csv
```
### Run Scripts
In the working directory, first run
```
python .\train_eval_split.py stores.csv [fraction]
```

*fraction*: the fraction of training dataset with respect to the whole dataset

For example, we could set fraction = 0.7 and we will get the train_data.csv with 140 data points and eval_data.csv with 60 data points:
```
python .\train_eval_split.py stores.csv 0.7
```

To train a local model, we could run:
```
source scripts/train-local.sh
```

To submit a Google AI platform job, we could run:
```
source scripts/train-cloud.sh
```
