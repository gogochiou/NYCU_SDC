#!/usr/bin/env python3

import random

def main():
    print("HW2-1")
    future_day = 10
    weight_list = {"sunny":[8, 2, 0],
                   "cloudy":[4, 4, 2],
                   "rainy":[2, 6, 2]}
    weather = ['sunny', 'cloudy', 'rainy']
    weather_list = ['sunny']

    for day in range(future_day):
        weight = weight_list[weather_list[day]]
        random_result = random.choices(weather,weights = weight)
        weather_list.append(random_result[0])
    
    print(weather_list)
        

if __name__ == "__main__":
    main()