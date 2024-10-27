import os

from palimpzest.datamanager import DataDirectory


def remove_cache():
    DataDirectory().clearCache(keep_registry=True)
    bad_files = [
        "testdata/enron-eval/assertion.log",
        "testdata/enron-eval/azure_openai_usage.log",
        "testdata/enron-eval/openai_usage.log",
    ]
    [os.remove(file) for file in bad_files if os.path.exists(file)]

    cache = DataDirectory().getCacheService()
    cache.rmCachedData("codeEnsemble")
    cache.rmCachedData("codeSamples")


def buildNestedStr(node, indent=0, buildStr=""):
    elt, child = node
    indentation = " " * indent
    buildStr = f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
    if child is not None:
        return buildNestedStr(child, indent=indent + 2, buildStr=buildStr)
    else:
        return buildStr


def get_models_from_physical_plan(plan) -> list:
    models = []
    while plan is not None:
        model = getattr(plan, "model", None)
        models.append(model.value if model is not None else None)
        plan = plan.source

    return models
