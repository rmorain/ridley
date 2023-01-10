# ridley
Knowledge integration demonstrated using riddle generation

### Environment

Create the conda environment from `environment.yml`
```
conda env create -f environment.yml
```



### Testing
Before merging, run the following to pass all tests:
```python -m unittest```

Also run the linter:

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
```
