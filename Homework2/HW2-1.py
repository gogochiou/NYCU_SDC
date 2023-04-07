#!/usr/bin/env python3

import random

def weather_distr(p_list_):
    distr = []
    for i in range(len(p_list_)):
        list_ = []
        for j in range(len(p_list_[i])):
            for k in range(p_list_[i][j]) :
                list_.append(j)
        random.shuffle(list_)
        distr.append(list_)
    return distr


def main():
    print("HW2-1")
    d_day = 10
    p_list = [[8, 2, 0],
              [4, 4, 2],
              [2, 6, 2]]
    w_distr = weather_distr(p_list)
    print(w_distr)

    weather_list = [0]
    for day in range(d_day):
        list_num = random.choice(range(10))
        last_w = weather_list[day]
        today_w = w_distr[last_w][list_num]
        weather_list.append(today_w)
    
    print(weather_list)
        

if __name__ == "__main__":
    main()