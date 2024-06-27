from __future__ import annotations
import json
import time
from typing import List, Tuple
from palimpzest.constants import Model
from palimpzest.datamanager.datamanager import DataDirectory
from .strategy import PhysicalOpStrategy

from palimpzest.utils import API, getJsonFromAnswer
from palimpzest import generators

from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, convert
from palimpzest.prompts import EXAMPLE_PROMPT, CODEGEN_PROMPT, ADVICEGEN_PROMPT

# TYPE DEFINITIONS
FieldName = str
CodeName = str
Code = str
DataRecordDict = Dict[str, Any]
Exemplar = Tuple[DataRecordDict, DataRecordDict]
CodeEnsemble = Dict[CodeName, Code]

class LLMConvertCodeSynthesis(convert.LLMConvert):

    code_strategy: CodeSynthStrategy # Default is CodeSynthStrategy.SINGLE,

    def __init__(self, 
                cache_across_plans: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_across_plans = cache_across_plans

        # read the list of exemplars already generated by this operator if present
        if self.cache_across_plans:
            cache = DataDirectory().getCacheService()
            exemplars_cache_id = self.get_op_id()
            exemplars = cache.getCachedData("codeExemplars", exemplars_cache_id)
            # set and return exemplars if it is not empty
            if exemplars is not None and len(exemplars) > 0:
                self.exemplars = exemplars
            else:
                self.exemplars = []
        else:
            self.exemplars = []
        self.field_to_code_ensemble = {}

    def _fetch_cached_code(self, fields_to_generate: List[str]) -> Tuple[Dict[CodeName, Code]]:
        # if we are allowed to cache synthesized code across plan executions, check the cache
        field_to_code_ensemble = {}
        cache = DataDirectory().getCacheService()
        for field_name in fields_to_generate:
            code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
            code_ensemble = cache.getCachedData("codeEnsembles", code_ensemble_cache_id)
            if code_ensemble is not None:
                field_to_code_ensemble[field_name] = code_ensemble

        # set and return field_to_code_ensemble if all fields are present and have code
        if all([field_to_code_ensemble.get(field_name, None) is not None for field_name in fields_to_generate]):
            self.field_to_code_ensemble = field_to_code_ensemble
            return self.field_to_code_ensemble
        else:
            return {}

    def _shouldSynthesize(self, 
                        exemplars: List[Exemplar],
                        num_exemplars: int=1,
                        code_regenerate_frequency: int=200,
                          *args, **kwargs) -> bool:
        """ This function determines whether code synthesis should be performed based on the strategy and the number of exemplars available. """
        raise NotImplementedError("This method should be implemented in a subclass")

    def _synthesize_field_code(
        self,
        api: API,
        output_field_name: str,
        code_ensemble_num: int=1,       # if strategy != SINGLE
        num_exemplars: int=1,           # if strategy != EXAMPLE_ENSEMBLE
    ) -> Tuple[Dict[CodeName, Code], StatsDict]:
        """ This method is responsible for synthesizing the code on a per-field basis. 
        Wrapping different calls to the LLM and returning a set of per-field query statistics.
        The format of the code ensemble dictionary is {code_name: code} where code_name is a string and code is a string representing the code.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def synthesize_code_ensemble(self, 
                                 fields_to_generate,
                                 candidate_dict: DataRecordDict, *args, **kwargs):
        """ This function is a wrapper around specific code synthesis methods 
        that wraps the synthesized code per-field in a dictionary and returns the stats object.
        """
        # synthesize the per-field code ensembles
        field_to_code_ensemble = {}
        output_stats = {}
        for field_name in fields_to_generate:
            api = API.from_input_output_schemas(
                inputSchema=self.inputSchema,
                outputSchema=self.outputSchema,
                field_name=field_name,
                input_fields=candidate_dict.keys()
            )

            # TODO here _synthesize_code should be called with the right parameters per-code-strategy?!
            code_ensemble, code_synth_stats = self._synthesize_field_code(api, field_name)
            field_to_code_ensemble[field_name] = code_ensemble

            for key,value in code_synth_stats.items():
                if type(value) == type(dict()):
                    for k2, v2 in value.items():
                        # TODO hacky bc we don't know the type of v2!
                        output_stats[k2] = output_stats.get(k2,type(v2)()) + v2
                else:
                    output_stats[key] = output_stats.get(key,type(value)()) + value

            # add code ensemble to the cache
            if self.cache_across_plans:
                cache = DataDirectory().getCacheService()
                code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
                cache.putCachedData("codeEnsembles", code_ensemble_cache_id, code_ensemble)

            # TODO: if verbose
            for code_name, code in code_ensemble.items():
                print(f"CODE NAME: {code_name}")
                print("-----------------------")
                print(code)

        # set field_to_code_ensemble and code_synthesized to True
        return field_to_code_ensemble, output_stats

    def _bonded_query_fallback(self, candidate):
        fields_to_generate = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)
        candidate_dict = candidate._asDict(include_bytes=False)

        prev_model = self.model
        prev_prompt_strategy = self.prompt_strategy

        # TODO: assert GPT-4 is available; maybe fall back to another model otherwise
        self.model = Model.GPT_4
        self.prompt_strategy = PromptStrategy.DSPY_COT_QA

        prompt = self._construct_query_prompt(fields_to_generate=fields_to_generate)
        text_content = json.dumps(candidate_dict)
        final_json_objects, query_stats = self._dspy_generate_fields(
            fields_to_generate=fields_to_generate,
            content=text_content,
            prompt=prompt
        )
        self.model = prev_model
        self.prompt_strategy = prev_prompt_strategy

        drs = []
        for idx, js in enumerate(final_json_objects):
            dr = self._create_data_record_from_json(
                jsonObj=js,
                candidate=candidate,
                cardinality_idx=idx
            )
            drs.append(dr)

        marshal_time = time.time() - self.start_time
        query_stats["total_time"] = marshal_time # TODO this is not correct
        # compute the record_op_stats for each data record and return
        record_op_stats_lst = self._create_record_op_stats_lst(records=drs, fields=fields_to_generate, query_stats=query_stats)

        # NOTE: this now includes bytes input fields which will show up as: `field_name = "<bytes>"`;
        #       keep an eye out for a regression in code synth performance and revert if necessary
        # update operator's set of exemplars
        exemplars = [dr._asDict(include_bytes=False) for dr in drs] # TODO: need to extend candidate to same length and zip
        self.exemplars.extend(exemplars)

        # if we are allowed to cache exemplars across plan executions, add exemplars to cache
        if self.cache_across_plans:
            cache = DataDirectory().getCacheService()
            exemplars_cache_id = self.get_op_id()
            cache.putCachedData(f"codeExemplars", exemplars_cache_id, exemplars)

        return drs, record_op_stats_lst

    def __call__(self, candidate):
        "This code is used for codegen with a fallback to default"
        self.start_time = time.time()

        fields_to_generate = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)
        # NOTE: the following is how we used to compute the candidate_dict; now that I am disallowing code synthesis for one-to-many queries, I don't think we need to invoke the _asJSONStr() method, which helped format the tabular data in the "rows" column for Medical Schema Matching. In the longer term, we should come up with a proper solution to make _asDict() properly format data which relies on the schema's _asJSONStr method.
        #   candidate_dict_str = candidate._asJSONStr(include_bytes=False, include_data_cols=False)
        #   candidate_dict = json.loads(candidate_dict_str)
        #   candidate_dict = {k: v for k, v in candidate_dict.items() if v != "<bytes>"}
        candidate_dict = candidate._asDict(include_bytes=False)

        # Check if code was already synthesized, or if we have at least one converted sample
        if self._shouldSynthesize():
            self.field_to_code_ensemble, total_code_synth_stats = (
                self.synthesize_code_ensemble(fields_to_generate, candidate_dict)
            )
            self.code_synthesized = True
        else:
            # read the dictionary of ensembles already synthesized by this operator if present
            if self.cache_across_plans:
                self.field_to_code_ensemble = self._fetch_cached_code(fields_to_generate)
            total_code_synth_stats = {}

        # if we have yet to synthesize code (perhaps b/c we are waiting for more exemplars),
        # use GPT-4 to perform the convert (and generate high-quality exemplars) using a bonded query
        if not len(self.field_to_code_ensemble):
            return self._bonded_query_fallback(candidate)

        # add total_code_synth_stats to query_stats
        query_stats = {}
        for key,value in total_code_synth_stats.items():
            if type(value) == type(dict()):
                for k2, v2 in value.items():
                    query_stats[k2] = query_stats.get(k2,0) + v2
            else:
                query_stats[key] = query_stats.get(key,type(value)()) + value

        # if we have synthesized code run it on each field
        field_outputs = {}
        for field_name in fields_to_generate:
            # create api instance for executing python code
            api = API.from_input_output_schemas(
                inputSchema=self.inputSchema,
                outputSchema=self.outputSchema,
                field_name=field_name,
                input_fields=candidate_dict.keys()
            )
            code_ensemble = self.field_to_code_ensemble[field_name]
            answer, exec_stats = generators.codeEnsembleExecution(api, code_ensemble, candidate_dict)

            if answer is not None:
                field_outputs[field_name] = answer
                for key, value in exec_stats.items():
                    query_stats[key] += value
            else:
                # if there is a failure, run a conventional query
                print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                prev_model = self.model
                prev_prompt_strategy = self.prompt_strategy
                self.model = Model.GPT_3_5
                self.prompt_strategy = PromptStrategy.DSPY_COT_QA
                prompt = self._construct_query_prompt(fields_to_generate=[field_name])
                text_content = json.dumps(candidate_dict)
                final_json_objects, field_stats = self._dspy_generate_fields(
                    fields_to_generate=[field_name],
                    content=text_content,
                    prompt=prmpy
                )
                self.model = prev_model
                self.prompt_strategy = prev_prompt_strategy

                # include code execution time in field_stats
                if "fn_call_duration_secs" not in field_stats:
                    field_stats["fn_call_duration_secs"] = 0.0
                field_stats["fn_call_duration_secs"] += exec_stats["fn_call_duration_secs"]

                # update query_stats
                for key, value in field_stats.items():
                    if type(value) == type(dict()):
                        for k2, v2 in value.items():
                            query_stats[k2] = query_stats.get(k2,type(v2)()) + v2
                    else:
                        query_stats[key] = query_stats.get(key,type(value)()) + value

                # NOTE: we disallow code synth for one-to-many queries, so there will only be
                #       one element in final_json_objects
                # update field_outputs
                field_outputs[field_name] = final_json_objects[0][field_name]

        drs = [
            self._create_data_record_from_json(
                jsonObj=field_outputs, candidate=candidate, cardinality_idx=0
            )
        ]

        # compute the record_op_stats for each data record and return
        record_op_stats_lst = self._create_record_op_stats_lst(
            drs, fields_to_generate, query_stats
        )

        return drs, record_op_stats_lst

class LLMConvertCodeSynthesisNone(LLMConvertCodeSynthesis):
    code_strategy = CodeSynthStrategy.NONE

    def _shouldSynthesize(self, *args, **kwargs):
        return False

    def _synthesize_field_code(self, api:API, *args, **kwargs):
        code = api.api_def() + "  return None\n"
        code_ensemble = {"{api.name}_v0": code}
        return code_ensemble, {}

class LLMConvertCodeSynthesisSingle(LLMConvertCodeSynthesis):
    code_strategy = CodeSynthStrategy.SINGLE

    def _shouldSynthesize(self, num_exemplars: int=1, *args, **kwargs) -> bool:
        """ This function determines whether code synthesis 
        should be performed based on the strategy and the number of exemplars available. """
        # The code is the same for ExampleEnsemble EXCEPT the >= should be strictly >
        # TODO is this intended or an oversight?
        return not self.code_synthesized and len(self.exemplars) >= num_exemplars

    def _code_synth_single(self, api: API, output_field_name: str, exemplars: List[Exemplar]=list(), advice: str=None, language='Python'):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'advice': f"Hint: {advice}" if advice else "",
        }
        prompt = CODEGEN_PROMPT.format(**context)
        print("PROMPT")
        print("-------")
        print(f"{prompt}")
        # TODO replace hardcoded call to the GPT-4 model
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        ordered_keys = [
            f'```{language}',
            f'```{language.lower()}',
            f'```'
        ]
        code = None
        for key in ordered_keys:
            if key in pred:
                code = pred.split(key)[1].split('```')[0].strip()
                break

        print("-------")
        print("SYNTHESIZED CODE")
        print("---------------")
        print(f"{code}")

        return code, stats

    def _synthesize_field_code(self, api:API, output_field_name:str, num_exemplars:int=1, *args, **kwargs):
        code, output_stats = self._code_synth_single(api, output_field_name, exemplars=self.exemplars[:num_exemplars])
        code_ensemble = {f"{api.name}_v0" : code}
        return code_ensemble, output_stats

# NOTE A nicer truly class based approach would re-implement the code_synth_single method with calls to __super__ and then only re-implement the differences instead of having the code in the superclass know about the subclass-specific parameters (i.e., advice).
class LLMConvertCodeSynthesisExampleEnsemble(LLMConvertCodeSynthesisSingle):
    code_strategy = CodeSynthStrategy.EXAMPLE_ENSEMBLE
    
    def _shouldSynthesize(self, num_exemplars: int=1, *args, **kwargs) -> bool:
        if len(self.exemplars) <= num_exemplars:
            return False
        return not self.code_synthesized

    def _synthesize_field_code(self, api:API, 
                         output_field_name:str, 
                         code_ensemble_num:int=1, *args, **kwargs):
        # creates an ensemble of `code_ensemble_num` synthesized functions; each of
        # which uses a different exemplar (modulo the # of exemplars) for its synthesis
        code_ensemble = {}
        output_stats = {}
        for i in range(code_ensemble_num):
            code_name = f"{api.name}_v{i}"
            exemplar = self.exemplars[i % len(self.exemplars)]
            code, stats = self._code_synth_single(api, output_field_name, exemplars=[exemplar])
            code_ensemble[code_name] = code
            for key,value in stats.items():
                if type(value) == type(dict()):
                    for k2, v2 in value.items():
                        output_stats[k2] = output_stats.get(k2,0) + v2 # Should we simply throw the usage away here?
                else:
                    output_stats[key] += output_stats.get(key,type(value)()) + value
        return code_ensemble, output_stats

class LLMConvertCodeSynthesisAdviceEnsemble(LLMConvertCodeSynthesisSingle):
    code_strategy = CodeSynthStrategy.ADVICE_ENSEMBLE

    def _shouldSynthesize(self, *args, **kwargs):
        return False

    def _parse_multiple_outputs(self, text, outputs=['Thought', 'Action']):
        data = {}
        for key in reversed(outputs):
            if key+':' in text:
                remain, value = text.rsplit(key+':', 1)
                data[key.lower()] = value.strip()
                text = remain
            else:
                data[key.lower()] = None
        return data

    def _synthesize_advice(self, 
                           api: API, 
                           output_field_name: str, 
                           exemplars: List[Dict[DataRecord, DataRecord]]=list(), language='Python', 
                           n_advices=4,
                           limit:int=3):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[0][field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[1][output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'n': n_advices,
        }
        prompt = ADVICEGEN_PROMPT.format(**context)
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        advs = self._parse_multiple_outputs(pred, outputs=[f'Idea {i}' for i in range(1, limit+1)])

        return advs, stats

    def _synthesize_field_code(self, 
                         api:API, 
                         output_field_name:str, 
                         code_ensemble_num:int=1,
                         num_exemplars: int = 1,
                         *args, **kwargs):
        # a more advanced approach in which advice is first solicited, and then
        # provided as context when synthesizing the code ensemble
        output_stats = {}
        # solicit advice
        advices, adv_stats = self._synthesize_advice(api, output_field_name, exemplars=self.exemplars[:num_exemplars], n_advices=code_ensemble_num)
        for key,value in adv_stats.items():
            if type(value) == type(dict()):
                for k2, v2 in value.items():
                    output_stats[k2] = output_stats.get(k2,0) + v2
            else:
                output_stats[key] += output_stats.get(key,type(value)()) + value

        code_ensemble = {}
        # synthesize code ensemble
        for i, adv in enumerate(advices):
            code_name = f"{api.name}_v{i}"
            code, stats = self._code_synth_single(api, output_field_name, exemplars=self.exemplars[:num_exemplars], advice=adv)
            code_ensemble[code_name] = code
            for key in output_stats.keys():
                output_stats[key] += stats[key]
        return code_ensemble, output_stats

class LLMConvertCodeSynthesisAdviceEnsembleValidation(LLMConvertCodeSynthesisSingle):
    code_strategy = CodeSynthStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION

    def _shouldSynthesize(self, code_regenerate_frequency:int = 200, *args, **kwargs):
        return len(self.exemplars) % code_regenerate_frequency == 0

    def _synthesize_field_code(self, api:API, output_field_name:str, exemplars:List[Exemplar]=list(), *args, **kwargs):
        # TODO this was not implemented ? 
        raise Exception("not implemented yet")

class CodeSynthesisConvertStrategy(PhysicalOpStrategy):
    """
    This strategy creates physical operator classes that convert records to one schema to another using code synthesis.

    """

    logical_op_class = logical.ConvertScan
    physical_op_class = LLMConvertCodeSynthesis

    code_strategy_map = {
        CodeSynthStrategy.NONE: LLMConvertCodeSynthesisNone,
        CodeSynthStrategy.SINGLE: LLMConvertCodeSynthesisSingle,
        CodeSynthStrategy.EXAMPLE_ENSEMBLE: LLMConvertCodeSynthesisExampleEnsemble,
        CodeSynthStrategy.ADVICE_ENSEMBLE: LLMConvertCodeSynthesisAdviceEnsemble,
        CodeSynthStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION: LLMConvertCodeSynthesisAdviceEnsembleValidation
    }

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy,
                code_synth_strategy: CodeSynthStrategy = CodeSynthStrategy.SINGLE,
                *args, **kwargs) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            # TODO restrict to only GPT 4 ? 
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            op_class = cls.code_strategy_map[code_synth_strategy]
            # physical_op_type = type(op_class.__name__+model.name,
            physical_op_type = type(op_class.__name__,
                                    (op_class,),
                                    {'model': model,
                                     'prompt_strategy': prompt_strategy,
                                     })
            return_operators.append(physical_op_type)

        return return_operators
