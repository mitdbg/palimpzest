import json
import time

import numpy as np
from palimpzest import dspyCOT, SingleQuestionOverSample
from fuzzywuzzy import process, fuzz

def get_hist_save(task, resolution=0.001):
    ranges = []
    with open("../data/test_v16_inputfile100-result-What is the aut-0.1-location.json", "rb") as file:
        data = file.read()
        json_obj = json.loads(data)

    for loc in json_obj["files"]:
        minv = loc["start"]/loc["total_chars"]
        maxv = loc["end"]/loc["total_chars"]
        print("min and max:", minv, maxv)
        ranges.append((minv, maxv))

    # Determine the overall min and max from the ranges for the heatmap extent
    overall_min = 0.0
    overall_max = 1.0

    # Create a frequency matrix based on the ranges
    index_range =  1/resolution
    bins = np.arange(overall_min, overall_max, resolution)
    frequency_matrix = np.zeros(len(bins)-1)

    # Populate the frequency matrix based on the ranges
    for r in ranges:
        start_index = int(float(r[0] - overall_min)/resolution)
        end_index = int(float(r[1] - overall_min)/resolution)
        frequency_matrix[start_index:end_index] += 1
    # save the index and the frequency_matrix to a csv file
    np.savetxt("../data/frequency-"+task+".csv",
               np.column_stack((bins[:-1] * index_range, frequency_matrix)),
               delimiter=",",
               fmt='%.3f')

def find_best_range(values, budget, trim_zeros=True):
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
    print ("best_start:", best_start, "best_end:", best_end)
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
        half_zeros = int((leading_zeros+trailing_zeros)/2)
        print("leading_zeros:", leading_zeros, "trailing_zeros:", trailing_zeros, "half_zeros:", half_zeros)
        best_start = best_start - half_zeros+leading_zeros
        best_end = best_end - trailing_zeros + leading_zeros+trailing_zeros-half_zeros

        if best_start < 0:
            best_end = best_end - best_start
            best_start = 0
        if best_end >= n:
            best_start = best_start - (best_end-n+1)
            best_end = n-1

    return best_start, best_end


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
    return  start *1.0/index_range, end *1.0/index_range

def get_test_result(file_path, question, sr, er):
    # Load document
    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)


    start = int(sr * test_len)
    end = int(er * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end+1]
    if sr == er:
        end = int((er+0.001) * test_len)-1
        sample = context[start:end+1]
    print("sample size:", len(sample))

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverSample)
    pred = cot(question, sample)
    return pred.answer

def run_file_batch(list_of_files, question, hist_file, budget=0.05):
    sr, er = get_range_from_hist(hist_file, budget, resolution=0.001, trim_zeros=False)
    print("start ratio:", sr, "end ratio:", er)

    results = {"question": question, "files": []}
    for file in list_of_files:
        start_time = time.time()
        test_result = get_test_result(file, question, sr, er)
        duration = time.time() - start_time

        results["files"].append({"file": file, "result": test_result, "duration": duration})
        print("file:", file, " result:", test_result)

    return results



def best_substring_match(query, string):
    # This will extract all substrings of length equal to the query from the string
    candidates = [string[i:i + len(query)] for i in range(len(string) - len(query) + 1)]
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
