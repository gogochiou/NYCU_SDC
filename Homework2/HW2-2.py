#!/usr/bin/env python3

import random

def main():
    print("HW2-2")
    future_day = 1000000
    weight_list = {"sunny":[8, 2, 0],
                   "cloudy":[4, 4, 2],
                   "rainy":[2, 6, 2]}
    weather = ['sunny', 'cloudy', 'rainy']
    weather_list = ['sunny']
    s_count = 0
    c_count = 0
    r_count = 0

    for day in range(future_day):
        weight = weight_list[weather_list[day]]
        random_result = random.choices(weather,weights = weight)
        weather_list.append(random_result[0])
        if random_result[0] == 'sunny':
            s_count+=1
        elif random_result[0] == 'cloudy':
            c_count+=1
        elif random_result[0] == 'rainy':
            r_count+=1
    
    # print(weather_list)
    print(s_count/future_day)
    print(c_count/future_day)
    print(r_count/future_day)

if __name__ == "__main__":
    main()