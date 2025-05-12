# MST_SE5211_CompInt
Final project for 2025 Spring SE5211 Computational Intelligence

Example usage
---
First, make sure you have the requirements `numpy`, `scikit-learn`, `scipy`, 'torch', 'torchmetrics', torch-geometric', 'ignite'. You can always install them with `pip`: `pip3 install -r requirements.txt`.

Then, to reproduce the DMoN results on the [Cora graph](https://relational.fel.cvut.cz/dataset/CORA), run

```python
python dmon_barebones.py --dataset='cora' --iteration=1 --clustmod=0
```
