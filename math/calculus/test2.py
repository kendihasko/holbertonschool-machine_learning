def summation_i_squared(n):
    sum = 0
    for i in range(1, n + 1):
        sum += i**2
        print(f"i ={1}, i^2 = {i**2}, sum = {sum}")

    return sum