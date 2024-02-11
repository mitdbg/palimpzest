from dsp.modules.hf import HFModel
import requests
import backoff

ERRORS = (Exception)

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )

class TogetherHFAdaptor(HFModel):
    def __init__(self, model, apiKey, **kwargs):
        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.api_base = "https://api.together.xyz/inference"
        self.token = apiKey
        self.model = model

        self.use_inst_template = False
        if any(keyword in self.model.lower() for keyword in ["inst", "instruct"]):
            self.use_inst_template = True

        stop_default = "\n\n---"

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1,
            "top_k": 20,
            "repetition_penalty": 1,
            "n": 1,
            "stop": stop_default if "stop" not in kwargs else kwargs["stop"],
            **kwargs
        }

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def _generate(self, prompt, use_chat_api=False, **kwargs):
        url = f"{self.api_base}"

        kwargs = {**self.kwargs, **kwargs}
        stop = kwargs.get("stop")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 150)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 50)
        repetition_penalty = kwargs.get("repetition_penalty", 1)
        prompt = f"[INST]{prompt}[/INST]" if self.use_inst_template else prompt

        if use_chat_api:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."},
                {"role": "user", "content": prompt}
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }
        else:
            #print("PROMPT", prompt)
            #print()
            #print()
            body = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }

        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            with self.session.post(url, headers=headers, json=body) as resp:
                resp_json = resp.json()
                #print("RESP JSON", resp_json)
                if use_chat_api:
                    completions = [resp_json['output'].get('choices', [])[0].get('message', {}).get('content', "")]
                else:
                    completions = [resp_json['output'].get('choices', [])[0].get('text', "")]
                response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
                return response
        except Exception as e:
            #print("EXCEPTION", e)
            if resp_json:
                print(f"resp_json:{resp_json}")
            print(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")

