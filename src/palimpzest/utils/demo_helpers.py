# for those of us who resist change and are still on Python 3.9
try:
    from itertools import pairwise
except Exception:
    from more_itertools import pairwise

import gradio as gr
import pandas as pd


def flatten_nested_tuples(nested_tuples):
    """
    This function takes a nested iterable of the form (4,(3,(2,(1,())))) and flattens it to (1, 2, 3, 4).
    """
    result = []

    def flatten(item):
        if isinstance(item, tuple):
            if item:  # Check if not an empty list
                flatten(item[0])  # Process the head
                flatten(item[1])  # Process the tail
        else:
            result.append(item)

    flatten(nested_tuples)
    result = list(result)
    result.reverse()
    return result[1:]


def create_plan_str(flatten_ops):
    """Helper function to return string w/physical plan."""
    plan_str = ""
    start = flatten_ops[0]
    plan_str += f" 0. {type(start).__name__} -> {start.output_schema.__name__} \n"

    for idx, (left, right) in enumerate(pairwise(flatten_ops)):
        in_schema = left.output_schema
        out_schema = right.output_schema
        plan_str += f" {idx+1}. {in_schema.__name__} -> {type(right).__name__} -> {out_schema.__name__} "
        if hasattr(right, "model"):
            plan_str += f"\n    Using {right.model}"
            if hasattr(right, "filter"):
                filter_str = (
                    right.filter.filter_condition
                    if right.filter.filter_condition is not None
                    else str(right.filter.filter_fn)
                )
                plan_str += f'\n    Filter: "{filter_str}"'
            if hasattr(right, "token_budget"):
                plan_str += f"\n    Token budget: {right.token_budget}"
        plan_str += "\n"
        plan_str += (
            f"    ({','.join(in_schema.field_names())[:15]}...) -> ({','.join(out_schema.field_names())[:15]}...)"
        )
        plan_str += "\n\n"

    return plan_str


def print_table(records, cols=None, plan_str=None):
    """Helper function to print execution results using Gradio"""
    if len(records) == 0:
        print("No records met search criteria")
        return

    records = [record.to_dict() for record in records]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols

    with gr.Blocks() as demo:
        gr.Dataframe(records_df[print_cols])

        if plan_str is not None:
            gr.Textbox(value=plan_str, info="Physical Plan")

    demo.launch()
