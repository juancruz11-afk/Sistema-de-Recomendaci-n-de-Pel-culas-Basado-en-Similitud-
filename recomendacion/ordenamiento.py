# ordenamiento.py
def quicksort_by_score(arr, key="score", reverse=False):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2][key]
    left = [x for x in arr if x[key] > pivot] if reverse else [x for x in arr if x[key] < pivot]
    middle = [x for x in arr if x[key] == pivot]
    right = [x for x in arr if x[key] < pivot] if reverse else [x for x in arr if x[key] > pivot]
    return quicksort_by_score(left, key, reverse) + middle + quicksort_by_score(right, key, reverse)
