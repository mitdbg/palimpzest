from black import Mode
from palimpzest.query.optimizer.optimizer import BayesianOptimizer
from pydantic import BaseModel, Field
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile
import palimpzest as pz
import litellm

from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.constants import Model
from palimpzest.validator.validator import Validator
import pandas as pd

if __name__ == "__main__":
    #litellm._turn_on_debug()
    # initial_dataset = [
    #     ("together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo", (0.7, 1.1, 0.5)),
    #     ("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", (0.85, 1.0, 0.8)),
    #     ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", (0.95, 1.8, 1.3)),
    #     ("together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", (0.87, 2.0, 2.0))
    # ]
    initial_dataset = [
        (Model.GPT_4_1, (1.0, 1.589207, 0.002878)),
        (Model.GPT_4_1_MINI, (1.0, 0.727763, 0.000474)),
        (Model.GPT_4_1_NANO, (1.0, 2.075557, 0.000163)),
        (Model.GPT_4o, (1.0, 2.642218, 0.004408)),
        (Model.GPT_4o_MINI, (1.0, 3.611015, 0.000242)),
        (Model.GPT_5, (1.0, 5.299906, 0.00412)),
        (Model.GPT_5_MINI, (1.0, 4.856572, 0.000696)),
        (Model.GPT_5_NANO, (1.0, 4.932934, 0.000210)),
        (Model.o4_MINI, (1.0, 2.306797, 0.002144))
    ]
   
    class EmailFile(BaseModel):
        filename: str = Field(description="The UNIX-style name of the file")
        contents: str = Field(description="The contents of the file")
        sender: str = Field(description="The email address of the email's sender")
        subject: str = Field(description="The subject of the email")
    email_idx = [9, 42, 45, 69]
    email_dataset = []
    for idx in email_idx:
        file_path = f"testdata/enron-eval-medium/allen-p-inbox-{idx}.txt"
        with open(file_path, 'r') as file:
            contents = file.read()
        email = DataRecord(data_item = TextFile(filename = f"allen-p-inbox-{idx}.txt", contents = contents), source_indices = f"{idx}")
        email_dataset.append(email)

    bo_optimizer = BayesianOptimizer(initial_dataset, pz.MinTime(), cost_budget=40, cost_model=None,
                                    input_schema=TextFile, output_schema=EmailFile, acq_func="EI",)
    best_model, best_posterior = bo_optimizer.optimize(email_dataset)
    print("Best model selected:", best_model)
    print("Best model posterior mean:", best_posterior.mean.item())
    print("Best model posterior variance:", best_posterior.variance.item())
    print("Final datapoints:")
    print("X:", bo_optimizer.X)
    print("X_models:", bo_optimizer.X_models)
    print("Y:", bo_optimizer.Y)

    # # Evaluate all models and save results to CSV
    # all_results = []
    # for model in Model:
    #     if not model.is_openai_model() or model.is_audio_model() or model.is_text_embedding_model():
    #         continue
    #     physical_op = LLMConvertBonded(model = model, input_schema = TextFile,
    #                 logical_op_id = "email_extracter", output_schema = EmailFile)
    #     for sample in email_dataset:
    #         single_result = {}
    #         data_record_set = physical_op(sample)
    #         output = {"sender": data_record_set.data_records[0].sender, "subject": data_record_set.data_records[0].subject}
    #         quality, gen_stats, full_hash = Validator(model = Model.GPT_5_NANO)._score_map(physical_op, fields = ["sender", "subject"],
    #                                                                 input_record = sample, output = output, full_hash="abc123")

    #         single_result['total_input_cost'] = data_record_set.record_op_stats[0].total_input_cost
    #         single_result['total_output_cost'] = data_record_set.record_op_stats[0].total_output_cost
    #         single_result['llm_call_duration_secs'] = data_record_set.record_op_stats[0].llm_call_duration_secs
    #         single_result['quality'] = quality
    #         single_result['model'] = model.value
    #         single_result['parent_ids'] = sample._source_indices[0]
    #         all_results.append(single_result)
    #     print(f"Completed model {model}")
    # df = pd.DataFrame(all_results)
    # print(df)
    # df.to_csv("gpt_email_results_good.csv", index=False)

