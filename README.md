# MST_SE5211_CompInt
Final project for 2025 Spring SE5211 Computational Intelligence.  Greetings Dr. Corns!

Example usage
---
First, make sure you have the requirements `numpy`, `scikit-learn`, `scipy`, 'torch', 'torchmetrics', torch-geometric'. You can always install them with `pip`: `pip3 install -r requirements.txt`.

Then, to reproduce the DMoN results on the [Cora graph](https://ieee-dataport.org/documents/cora), run

```python
python3 dmon_barebones.py --dataset='cora' --iteration=1 --clustmod=0
```

'dataset' options can be found in dmon_barebones.py.

'iteration' is simply a counter used to append to output filenames.

'clustmod' is a scalar to manually change the number of clusters that the algorithm calculates.  '0' means that the number of labels given in the dataset file will be used.
