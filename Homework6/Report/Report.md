# **HW6 Report**

> ID : A111137
>
> Name : 邱柏鎧

## **Part 1. Plot Data**

- Green Line : History trajectory

- Red Line : Future trajectory

- White point : Lane Center

- White polygon : Lane Border

- Yellow polygon : Crosswalk polygon

- Blue line : Trajectories of surronding objects

![visualization](./Picture/visualization.png)

## **Part 2. Motion Prediction Model**

### **a. Baseline Model**

#### **- Model structure**

![Structure](./Picture/NNstruct.png)

#### **- Result**

<!-- ![Success](./Picture/Baseline_works.png) -->

<img src=./Picture/Baseline_works.png width=85%>

#### **- Failure Case study**

In the basic condition like driving straight forward, the model seems predicting well. However, if the car intend to cornering, the final result may looks not realiable, nevertheless, the tendency still works well in turning condition. (Not all 6 prediction works well, some of them will diverge!)

Belows are some failure condition :

> Green line with high transparent is groundtruth

- Changing lane

    ![Fail - ChangeLane](./Picture/Baseline_changinglaneFailure.png)

- Cornering

    ![Fail - Cornering](./Picture/Baseline_corneringFailure.png)

- Unsure Failure (guess : acceleration from static condition)

    ![Fail - unsure](./Picture/Baseline_accFailure.png)

    ![Fail - unsure](./Picture/Baseline_notsureFailure.png)

### **b. Modify Model**

#### **- Model structure**

To avoid problem of some condition, like changing lane problem of cornering problem, we need to consider surrounding object (neighbor). But not all object need to be consider, so self-attention can accomplish this requirement. Using exist multihead attention structure, we can easily reach the goal.

![Structure](./Picture/Modify_NNstruct.png)

#### **- Result**

- Straight forward

    ![straight forward](./Picture/Modify_model/Predict_works.png)

- Cornering

    ![cornering](./Picture/Modify_model/Cornering_better.png)

- Acceleration

    ![acc](./Picture/Modify_model/acc_seemsSOSO.png)

- Lane Change

    ![laneChange](./Picture/Modify_model/LaneChange_works.png)

#### **- Failure Case study**

Some of the condition still not works well, cornering and acceleration(deceleration) conditions are hard to predicted by just adding neighbor into nn structure. Sometimes it works, but can't be 100% guaranty. The best way is changing some structure in neural network, or even changing the design of loss function.

![corneringFailure](./Picture/Modify_model/Cornering_failure.png)

## **Part 3. Validation on KungFu road data**

### **- Result**

- Straight forward

    ![straight](./Picture/KungFuResult/Straight3.png)

- Cornering

    ![cornering](./Picture/KungFuResult/Cornering.png)

- Accelearation

    ![acc](./Picture/KungFuResult/acc.png)

- Deceleration

    ![deceleration](./Picture/KungFuResult/deceleration.png)

### **- Failure Case study**

Some of the condition works unacceptable. Changing lane and start from stop condition are hard to predict in my training network. In my opinion, maybe lane-changing is not rarely based on neighbors' motion, the traffic light, or some action policy may make prediction more difficult.

![fail](./Picture/KungFuResult/ChangeLane_Failure.png)