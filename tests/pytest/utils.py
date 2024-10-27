import os

from palimpzest.datamanager import DataDirectory


def remove_cache():
    DataDirectory().clear_cache(keep_registry=True)
    bad_files = [
        "testdata/enron-eval/assertion.log",
        "testdata/enron-eval/azure_openai_usage.log",
        "testdata/enron-eval/openai_usage.log",
    ]
    [os.remove(file) for file in bad_files if os.path.exists(file)]

    cache = DataDirectory().get_cache_service()
    cache.rm_cached_data("codeEnsemble")
    cache.rm_cached_data("codeSamples")


def build_nested_str(node, indent=0, build_str=""):
    elt, child = node
    indentation = " " * indent
    build_str = f"{indentation}{elt}" if indent == 0 else build_str + f"\n{indentation}{elt}"
    if child is not None:
        return build_nested_str(child, indent=indent + 2, build_str=build_str)
    else:
        return build_str


def get_models_from_physical_plan(plan) -> list:
    models = []
    while plan is not None:
        model = getattr(plan, "model", None)
        models.append(model.value if model is not None else None)
        plan = plan.source

    return models
