import os
import datasets
print("datasets path:", datasets.__file__)
print(os.listdir(os.path.dirname(datasets.__file__)))
