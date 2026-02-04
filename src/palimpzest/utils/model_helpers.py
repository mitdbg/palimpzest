import os

from palimpzest.constants import MAX_AVAILABLE_MODELS, Model
from palimpzest.core.models import PlanCost
from palimpzest.policy import Policy


def get_models(include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None) -> list[Model]:
    """
    Return the set of models which the system has access to based on the set environment variables.
    """
    models = []
    all_models = Model.get_all_models()

    if os.getenv("OPENAI_API_KEY") not in [None, ""]:
        openai_models = [model for model in all_models if model.is_provider_openai()]
        if not include_embedding:
            openai_models = [
                model for model in openai_models if not model.is_embedding_model()
            ]
        models.extend(openai_models)

    if os.getenv("TOGETHER_API_KEY") not in [None, ""]:
        together_models = [model for model in all_models if model.is_provider_together_ai()]
        if not include_embedding:
            together_models = [
                model for model in together_models if not model.is_embedding_model()
            ]
        models.extend(together_models)

    if os.getenv("ANTHROPIC_API_KEY") not in [None, ""]:
        anthropic_models = [model for model in all_models if model.is_provider_anthropic()]
        if not include_embedding:
            anthropic_models = [
                model for model in anthropic_models if not model.is_embedding_model()
            ]
        models.extend(anthropic_models)

    gemini_credentials_path = (
        os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
        if gemini_credentials_path is None
        else gemini_credentials_path
    )
    if os.getenv("GEMINI_API_KEY") not in [None, ""] or (use_vertex and os.path.exists(gemini_credentials_path)):
        vertex_models = [model for model in all_models if model.is_provider_vertex_ai()]
        google_ai_studio_models = [model for model in all_models if model.is_provider_google_ai_studio()]
        if not include_embedding:
            vertex_models = [
                model for model in vertex_models if not model.is_embedding_model()
            ]
            google_ai_studio_models = [
                model for model in google_ai_studio_models if not model.is_embedding_model()
            ]
        if use_vertex:
            models.extend(vertex_models)
        else:
            models.extend(google_ai_studio_models)

    return models

def get_optimal_models(policy: Policy, include_embedding: bool = False, use_vertex: bool = False, gemini_credentials_path: str | None = None) -> list[Model]:
    """
    Selects the top models from the available list based on the user's policy.

    Post-condition: This function will never return an empty list unless there are
    no available models at all. If policy constraints filter out all models, it
    falls back to returning the best model(s) based on the policy's primary metric.
    """
    # gather available models
    available_models = get_models(
        include_embedding=include_embedding,
        use_vertex=use_vertex,
        gemini_credentials_path=gemini_credentials_path,
    )

    if not available_models:
        return []

    # gather metrics for all models
    all_model_metrics = []
    for model in available_models:
        quality_score = model.get_overall_score()
        cost = model.get_usd_per_output_token()
        time_val = model.get_seconds_per_output_token()

        if quality_score is None:
            quality_score = 0
        if cost is None:
            cost = float("inf")
        if time_val is None:
            time_val = float("inf")

        all_model_metrics.append({
            "id": model,
            "quality": quality_score,
            "cost": cost,
            "time": time_val
        })

    # apply constraints
    candidates = []
    for model_data in all_model_metrics:
        normalized_quality = model_data["quality"] / 100.0
        proxy_plan = PlanCost(cost=0.0, time=0.0, quality=normalized_quality)

        if policy.constraint(proxy_plan):
            candidates.append(model_data)

    # fallback: If no models meet constraints, select best model(s) by primary metric
    if not candidates:
        primary_metric = policy.get_primary_metric()

        if primary_metric == "quality":
            # return the model with the highest quality score
            best = max(all_model_metrics, key=lambda x: x["quality"])
        elif primary_metric == "cost":
            # return the model with the lowest cost
            best = min(all_model_metrics, key=lambda x: x["cost"])
        elif primary_metric == "time":
            # return the model with the lowest latency
            best = min(all_model_metrics, key=lambda x: x["time"])
        else:
            # default to highest quality
            best = max(all_model_metrics, key=lambda x: x["quality"])

        return [best["id"]]

    # normalize metrics using min-max normalization
    quals = [c["quality"] for c in candidates]
    costs = [c["cost"] for c in candidates]
    times = [c["time"] for c in candidates]

    min_q, max_q = min(quals), max(quals)
    min_c, max_c = min(costs), max(costs)
    min_t, max_t = min(times), max(times)

    def normalize(val, min_v, max_v, invert=False):
        if max_v == min_v:
            return 1.0
        norm = (val - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    # get weight for each metric based on policy
    weights = policy.get_dict()
    w_q = weights.get("quality", 0.0)
    w_c = weights.get("cost", 0.0)
    w_t = weights.get("time", 0.0)

    scored_candidates = []
    for cand in candidates:
        n_q = normalize(cand["quality"], min_q, max_q, invert=False)
        n_c = normalize(cand["cost"], min_c, max_c, invert=True)
        n_t = normalize(cand["time"], min_t, max_t, invert=True)

        score = (w_q * n_q) + (w_c * n_c) + (w_t * n_t)

        scored_candidates.append((score, cand["id"]))

    # select the top-k candidates based on score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_models = [model for _, model in scored_candidates[:MAX_AVAILABLE_MODELS]]

    return top_models

def resolve_reasoning_settings(model: Model | None, reasoning_effort: str | None) -> tuple[bool, str]:
    """
    Resolve the reasoning settings based on the model and provided reasoning effort.
    Returns a tuple indicating whether reasoning prompt should be used and the reasoning effort level.
    By default, we use the reasoning prompt everywhere while setting the model reasoning effort to None (or minimal).
    If a user explicitly provides a reasoning_effort, we pass that through to the model.
    If the user explicitly disables reasoning_effort, we disable the reasoning prompt as well.
    """
    # turn off reasoning prompt if reasoning_effort is in [None, "disable", "minimal", "low"]
    use_reasoning_prompt = reasoning_effort not in [None, "disable", "minimal", "low"]

    # if reasoning_effort is set to "default", set it to None to use model defaults
    if reasoning_effort == "default":
        reasoning_effort = None

    # translate reasoning_effort into model-specific settings
    if model is not None and model.is_reasoning_model():
        if model.is_provider_vertex_ai() or model.is_provider_google_ai_studio():
            if reasoning_effort is None and model in [Model.GEMINI_2_5_PRO, Model.GOOGLE_GEMINI_2_5_PRO]:
                reasoning_effort = "low"
            elif reasoning_effort is None:
                reasoning_effort = "disable"
        elif model.is_provider_openai():
            reasoning_effort = "minimal" if reasoning_effort in [None, "disable", "minimal", "low"] else reasoning_effort

    return use_reasoning_prompt, reasoning_effort
