# Solving Linear Programs with GNN
In the file https://github.com/wellington36/Solving_Linear_Programs_with_GNN/tree/main/data-testing we have the testing set with some examples of structure (generates by https://github.com/liujl11git/GNN-LP/blob/main/1_generate_data.py). In the file https://github.com/wellington36/Solving_Linear_Programs_with_GNN/tree/main/salved_models we have some trained models (generates by https://github.com/liujl11git/GNN-LP/blob/main/2_training.py with different parameters). We can check the mse in test with the follow command:

```python
python 4_testing_all.py --type obj --set test --loss mse
```

Out:
```
MODEL: ./saved-models/obj_d500_s6.pkl, DATA-SET: ./data-testing/, NUM-DATA: 1000, LOSS: mse, ERR: 0.3851677179336548
MODEL: ./saved-models/obj_d100_s6.pkl, DATA-SET: ./data-testing/, NUM-DATA: 1000, LOSS: mse, ERR: 0.2846706509590149
MODEL: ./saved-models/obj_d2500_s6.pkl, DATA-SET: ./data-testing/, NUM-DATA: 1000, LOSS: mse, ERR: 0.17373700439929962
```
