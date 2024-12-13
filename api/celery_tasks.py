from celery import shared_task

from api.calc.fib import fib

# Example task
@shared_task
def process_data(file_path: str):
    print(f"Processing file at {file_path}")
    # Simulate long task
    with open(file_path, "r") as file:
        content = file.read()
    print(f"File content: {content}")
    return f"Processed {file_path}"

@shared_task
def dummy_test(num: int):
    print("Starting a dummy process...")
    # Simulate long task
    result = fib(num)
    return f"The {num}th fibonacci number is {result}"
