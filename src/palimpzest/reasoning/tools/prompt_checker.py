from palimpzest.reasoning.action_types import ExecuteAction
from palimpzest.reasoning.eval_response import EvaluationResponse
from palimpzest.reasoning.llm_api.llm_api import LLMClient
import json

class PromptChecker:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm_client = LLMClient(model=model)

    def extract_datasource_annotation(self, question: str) -> dict[str, str]:
        return {
            "fields": ["sender", "subject"],
            "examples": [{
                "datasource": """Message-ID: <1390685.1075853083264.JavaMail.evans@thyme>\nDate: Mon, 17 Sep 2001 07:56:52 -0700 (PDT)\nFrom: steven.january@enron.com\nTo: shelley.corman@enron.com, lynn.blair@enron.com, rick.dietz@enron.com, \n\tbradley.holmes@enron.com, donna.scott@enron.com, \n\tmike.bryant@enron.com, sharon.brown@enron.com, \n\tdarrell.schoolcraft@enron.com, gary.spraggins@enron.com, \n\tdale.ratliff@enron.com, ricki.winters@enron.com\nSubject: VACATION\nMime-Version: 1.0\nContent-Type: text/plain; charset=us-ascii\nContent-Transfer-Encoding: 7bit\nX-From: January, Steven </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SJANUARY>\nX-To: Corman, Shelley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Scorman>, Blair, Lynn </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Lblair>, Dietz, Rick </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rdietz>, Holmes, Bradley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Bholmes>, Scott, Donna </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dscott1>, Bryant, Mike </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Mbryant>, Brown, Sharon </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Sbrown1>, Schoolcraft, Darrell </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dschool>, Spraggins, Gary </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Gspragg>, Ratliff, Dale </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dratlif>, Winters, Ricki </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rwinter>\nX-cc: \nX-bcc: \nX-Folder: \\LBLAIR (Non-Privileged)\\Blair, Lynn\\Vacations 2001\nX-Origin: Blair-L\nX-FileName: LBLAIR (Non-Privileged).pst\n\nI plan on taking vacation October 4, 5, 8,9 10, 11, and 12. This will finish my vacation for the year. thanks. sj""",
                "annotated_fields": {
                    "sender": "steven.january",
                    "subject": "Vacation"
                }
            },{
                "datasource": """Message-ID: <760490.1075853083334.JavaMail.evans@thyme>
Date: Fri, 6 Jul 2001 11:17:00 -0700 (PDT)
From: sheila.nacey@enron.com
To: lynn.blair@enron.com, mike.bryant@enron.com, shelley.corman@enron.com, 
	rick.dietz@enron.com, bradley.holmes@enron.com, 
	steven.january@enron.com, donna.scott@enron.com, 
	gina.taylor@enron.com
Subject: Vacation plans
Cc: ricki.winters@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
Bcc: ricki.winters@enron.com
X-From: Nacey, Sheila </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SNACEY>
X-To: Blair, Lynn </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Lblair>, Bryant, Mike </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Mbryant>, Corman, Shelley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Scorman>, Dietz, Rick </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rdietz>, Holmes, Bradley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Bholmes>, January, Steven </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Sjanuary>, Scott, Donna </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dscott1>, Taylor, Gina </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Gtaylor10>
X-cc: Winters, Ricki </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rwinter>
X-bcc: 
X-Folder: \LBLAIR (Non-Privileged)\Blair, Lynn\Vacations 2001
X-Origin: Blair-L
X-FileName: LBLAIR (Non-Privileged).pst

I will be taking Monday, 7/16, off as a day of vacation.  Also, I have scheduled 8/24 thru 31 and 10/29 thru 11/5.  Almost afraid to go anywhere!  Sheila""",
                "annotated_fields": {
                    "sender": "sheila.nacey",
                    "subject": "Vacation Plans"
                }
            }]
        }
    
    def extract_answer(self, result: str) -> dict:
        result = result.replace("```json", "").replace("```", "")
        result = result.strip()
        return json.loads(result)
    
    def evaluate_prompt(self, question: str, step_action: ExecuteAction) -> EvaluationResponse:
        prompt_template = step_action.answer
        prompt_template = prompt_template.replace("{", "{{").replace("}", "}}")
        prompt_template = prompt_template.replace("{{datasource}}", "{datasource}")
        examples = self.extract_datasource_annotation(question)
    
        evaluation_response = EvaluationResponse()
        evaluation_response.pass_ = True
        for example in examples["examples"]:
            try:
                prompt = prompt_template.format(datasource=example["datasource"])
                response = self.llm_client.get_completion(prompt)
                answer = self.extract_answer(response)

                evaluation_response.think += f"Given an example: {example['datasource']}\n"
                evaluation_response.think += f"The response from LLM: {response}\n"
                evaluation_response.think += f"After using JSON format extracting the fields, the answer is: {answer}. Then we can see:\n"
                for field in example["annotated_fields"]:
                    if field in answer:
                        print(answer[field], example["annotated_fields"][field])
                        if answer[field] != example["annotated_fields"][field]:
                            evaluation_response.pass_ = False
                            evaluation_response.think += f"Field {field} is incorrect. Expected {example['annotated_fields'][field]}, but got {answer[field]}.\n"
                        else:
                            evaluation_response.pass_ = evaluation_response.pass_ and True
                            evaluation_response.think += f"Field {field} is correct.\n"
                    else:
                        evaluation_response.pass_ = False
                        evaluation_response.think += f"Field {field} is not in the answer.\n"
                evaluation_response.think += "\n\n"
            except KeyError as e:
                evaluation_response.pass_ = False
                evaluation_response.think += f"Error formatting prompt template: {e}. The template may contain placeholders that don't match the available data. We only allow datasource as the placeholder.\n"
            except Exception as e:
                evaluation_response.pass_ = False

        print(evaluation_response.think)
        return evaluation_response
