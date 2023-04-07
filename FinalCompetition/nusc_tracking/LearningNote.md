# **Learning Note**

> **Nuscenes (testing data)** -- [Ranking and Thesis](https://www.nuscenes.org/tracking/?externalData=all&mapData=all&modalities=Any)

## **Method 1 : Global greedy**

### **Reference**

1. [MOT 的 Data Association](https://blog.51cto.com/u_15221047/2807357)

2. [目标跟踪算法 —— IoU Tracker 和 V-IoU Tracker](https://blog.csdn.net/BeBuBu/article/details/107227289)

3. [A Two-Stage Data Association Approach for 3D Multi-Object Tracking](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8122257/)

### **Code Explaination**

### **Result**

Eval time: 2979.4s

| GRADE | VALUE | GRADE | VALUE |
| ---- | ---- | ---- | ---- |
| **AMOTA** | 0.667 | **FAF** | 44.8 |
| **AMOTP** | 0.528 | **TP** | 79745 |
| **RECALL** | 0.697 | **FP** | 12596 |
| **MOTAR** | 0.817 | **FN** | 21574 |
| **GT** | 14556 | **IDS** | 578 |
| **MOTA** | 0.572 | **FRAG** | 358 |
| **MOTP** | 0.332 | **TID** | 0.40 |
| **MT** | 4333 | **LGD** | 0.76 |
| **ML** | 1685 |

## **Method 2 : CMBOT (Local greedy)**

### **Reference**

1. [Score refinement for confidence-based 3D multi-object tracking](https://github.com/cogsys-tuebingen/CBMOT)

2. [CenterTrack](https://github.com/xingyizhou/CenterTrack)

3. [CenterPointDetection](https://github.com/tianweiy/CenterPoint)

### **Code Explaination**

### **Result**

Eval time: 2921.4s

| GRADE | VALUE | GRADE | VALUE |
| ---- | ---- | ---- | ---- |
| **AMOTA** | 0.680 | **FAF** | 42.9 |
| **AMOTP** | 0.530 | **TP** | 79461 |
| **RECALL** | 0.694 | **FP** | 12060 |
| **MOTAR** | 0.822 | **FN** | 21988 |
| **GT** | 14556 | **IDS** | 448 |
| **MOTA** | 0.575 | **FRAG** | 327 |
| **MOTP** | 0.334 | **TID** | 0.35 |
| **MT** | 4307 | **LGD** | 0.72 |
| **ML** | 1745 |