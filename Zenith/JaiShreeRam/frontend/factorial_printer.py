def calculate_factorial(n):
    """
    Calculates the factorial of a non-negative integer n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

if __name__ == "__main__":
    number = 5
    try:
        fact_result = calculate_factorial(number)
        print(f"The factorial of {number} is: {fact_result}")
    except ValueError as e:
        print(f"Error: {e}")
