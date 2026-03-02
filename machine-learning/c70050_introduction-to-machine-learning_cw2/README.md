# How to run the code for Part 2


## How to test the regressor on the secret dataset

In order to test the regressor on the secret dataset, modify the main function to:
```python
if __name__ == "__main__":
  # Here you add the path to the dataset that you want to use.
  test_regressor_on_dataset(PATH_TO_DATASET) 
```


## How to train regressor on a dataset

To train a regressor on a given dataset, modify the main function to:
```python
if __name__ == "__main__":
  # Here you add the path to the dataset that you want to use.
  regressor_on_dataset(PATH_TO_DATASET)
```
The dataset is split into train and test. The model is trained using the train batch and then evaluated on the test batch and the whole dataset.

In order to modify the `train:val` ratio, modify the `data.sample()` function from `regressor_on_dataset()`.


## How to use hyperparameter search on a dataset

To use hyperparameter search on a dataset, modify the main function to:
```python
if __name__ == "__main__":
  # Here you add the path to the dataset that you want to use.
  regressor_hyperparameter_search_on_dataset(PATH_TO_DATASET)
```
Running will split the dataset into train, validation and test and then perform the hyperparameter search using the train and validation parts. The best model will then be evaluated on the test batch and the whole dataset.

In order to modify the `train:val:test` ratio, modify the `np.split()` function from `regressor_hyperparameter_search_on_dataset()`.