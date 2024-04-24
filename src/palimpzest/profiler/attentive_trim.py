from fuzzywuzzy import process, fuzz


def find_best_range(values, budget, trim_zeros=False):
    """
    Finds the consecutive range with the biggest sum within a budget.

    Args:
        values: A list of non-negative numbers.
        budget: The maximum number of consecutive elements to consider.

    Returns:
        A tuple containing the start and end indices (inclusive) of the best range,
        or None if the array is empty.
    """
    if not values:
        return None

    n = len(values)
    best_sum, best_start, current_sum, current_start = 0, 0, 0, 0

    # Iterate through the array, keeping track of current and best ranges.
    for i in range(n):
        current_sum += values[i]

        # If the current range exceeds the budget, remove elements from the beginning.
        while current_start + budget - 1 < i and current_start + budget - 1 >= 0:
            current_sum -= values[current_start]
            current_start += 1

        # Update best range if the current sum is bigger.
        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start

    best_end = best_start + budget - 1
    print("best_start:", best_start, "best_end:", best_end)
    if trim_zeros:
        # Trim leading/trailing zeros
        while best_start >= 0 and values[best_start] == 0:
            best_start += 1

        while best_end < n and values[best_end] == 0:
            best_end -= 1
    else:
        # balance the zero entries equally on both sides
        leading_zeros = 0
        trailing_zeros = 0
        start_idx = best_start
        end_idx = best_end
        while start_idx >= 0 and values[start_idx] == 0:
            leading_zeros += 1
            start_idx += 1
        while end_idx < n and values[end_idx] == 0:
            trailing_zeros += 1
            end_idx -= 1
        half_zeros = int((leading_zeros + trailing_zeros) / 2)
        print("leading_zeros:", leading_zeros, "trailing_zeros:", trailing_zeros, "half_zeros:", half_zeros)
        best_start = best_start - half_zeros + leading_zeros
        best_end = best_end - trailing_zeros + leading_zeros + trailing_zeros - half_zeros

        if best_start < 0:
            best_end = best_end - best_start
            best_start = 0
        if best_end >= n:
            best_start = best_start - (best_end - n + 1)
            best_end = n - 1

    return best_start, best_end + 1


def get_range_from_hist(file_path, range_budget, resolution=0.001, trim_zeros=True):
    # Load data from csv file and extract he second column as values
    values = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            values.append(int(float(line.split(",")[1])))
    index_range = 1 / resolution
    budget = int(range_budget * index_range)
    # Find the best range
    start, end = find_best_range(values, budget, trim_zeros=trim_zeros)
    print("start:", start, "end:", end, "index_range:", index_range)
    return start * 1.0 / index_range, end * 1.0 / index_range


def get_trimed(context, sr, er):
    test_len = len(context)
    start = int(sr * test_len)
    end = int(er * test_len)
    print("character start:", start, "end:", end)
    sample = context[start:end]
    return sample


# update the heatmap json object by increase the counter and refresh the heat region based on the new start and end index
#                 json_object = {'prompt_schema': prompt_schema,
#                                'question': question,
#                                'resolution': TOKEN_REDUCTION_GRANULARITY,
#                                'count': 0,
#                                'heatmap': {hist}}
def update_heatmap_json(j_obj, si, ei):
    j_obj["count"] += 1
    # iterate from si to ei in heatmap array and increase the counter
    for i in range(si, ei):
        # now we add all the count by 1, we may consider the variable weight based result quality in the future
        j_obj["heatmap"][i] += 1
    return j_obj


def best_substring_match(query, context):
    # This will extract all substrings of length equal to the query from the string
    candidates = [context[i:i + len(query)] for i in range(len(context) - len(query) + 1)]
    print("grd:", query)
    # Find the best match among the candidates
    ret = process.extractOne(query, candidates, scorer=fuzz.ratio)
    if ret is None:
        return None

    best_match, score = ret
    positions = [can == best_match for can in candidates]
    start = positions.index(True)
    end = start + len(query)
    print("best match:", best_match, "score:", score, "start:", start, "end:", end)
    # print("-------", string[start:end])
    return start, end
