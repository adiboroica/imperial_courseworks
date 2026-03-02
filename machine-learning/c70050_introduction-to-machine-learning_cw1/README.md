# Decision Tree Coursework

## Setting up the app

```bash
python3 -m venv venv            # Create the virtual environment 
source venv/bin/activate/       # Activate the virtual environment 
pip install --upgrade pip       # Install/Upgrade pip  
pip install -r requirements.txt # Install the requirements 
```

## Run the tests

```bash
python3 -m unittest                 # Run all the tests
python3 -m test.<path-to-test-file> # Run a certain test
(for example python3 -m unittest test.dataset.test_entropy)
```

## Running app on secret data
1. Clone the repository.
2. Add secret dataset to the repository.
3. Set DATA_PATH variable in main.py to the path to the secret dataset.
4. Run main.py
```bash
python3 main.py
```
