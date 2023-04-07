# **Record**

## **Code explain**

### **a). list comprehension**

Create a new list object from the items in **iterable**, filtered by **if condition**

```python
# General expression
[item for item in iterable if condition]
# In dataset.py code
elem = [k for k, v in batch[0].items()]
```

> Reference :
>
> 1. [stack overflow question](https://stackoverflow.com/questions/26536042/what-is-an-expression-such-as-dk-for-k-in-d-called)