import re
from datetime import datetime


def remove_extra_line_breaks(text: str) -> str:
    """Remove extra line breaks from the text"""
    return re.sub(r'\n{2,}', '\n\n', text)

def orchestrator_prompt(
    diary_context: list[str] | None = None,
    original_question: str = "",
    current_question: str = "",
    bad_context: list[dict] | None = None,
    allow_reflect: bool = True,
    allow_answer: bool = True,
    all_knowledge: list[str] | None = None,
) -> str:
    sections: list[str] = []
    action_sections: list[str] = []
    new_line = "\n"

    # Add header section
    sections.append("""You are an advanced AI agent. You are specialized in multistep reasoning. 
Using your training data and prior lessons learned, fullfill user's requests.
""")
    
    if original_question == "" or current_question == "":
        return ""
    
    if original_question == current_question:
        sections.append(f""" The original question is:
<original-question>
{original_question}
</original-question>
""")
    else:
        sections.append(f""" The current question is:
<current-question>
{current_question}
</current-question>
""")

    if diary_context:
        sections.append(f"""
    You have conducted the following actions:
    <context>
    {new_line.join(diary_context)}

    </context>
    """)

    # Add bad context section if exists
    if bad_context:
        attempts = []
        for i, c in enumerate(bad_context, 1):
            attempt = f"""
<attempt-{i}>
- Question: {c['question']}
- Answer: {c['answer']}
- Reject Reason: {c['evaluation']}
- Actions Recap: {c['error_analysis']['recap']}
- Actions Blame: {c['error_analysis']['blame']}
</attempt-{i}>"""
            attempts.append(attempt)

        learned_strategy = '\n'.join(c['error_analysis']['improvement_plan'] for c in bad_context)

        sections.append(f"""
Also, you have tried the following actions but failed to find the answer to the question:
<bad-attempts>    

{(new_line + new_line).join(attempts)}

</bad-attempts>

Based on the failed attempts, you have learned the following strategy:
<learned-strategy>
{learned_strategy}
</learned-strategy>
""")

    if all_knowledge:
        sections.append(f"""
This is the pre-knowledge you have learned from previous conversations:
<all-knowledge>
{new_line.join(all_knowledge)}
</all-knowledge>
""")

# - For greetings, casual conversation, general knowledge questions answer directly without references.
# - If user ask you to retrieve previous messages or chat history, remember you do have access to the chat history, answer directly without references.
# - For all other questions, provide a verified answer with references. Each reference must include exactQuote, url and datetime.
    if allow_answer:
        action_sections.append("""
<action-answer>
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- If uncertain, use <action-reflect>
- The returned format must with the following fields:
  - action: The action you suggest to take as next step, action-answer or action-reflect
  - answer: The answer to the question
</action-answer>
""")

    if allow_reflect:
        action_sections.append("""
<action-reflect>
- Critically examine <question>, <context>, <datasource>, <bad-attempts>, and <learned-strategy> to identify gaps and the problems. 
- Identify gaps and ask key clarifying questions that deeply related to the original question and lead to the answer
- Ensure each reflection:
 - Cuts to core emotional truths while staying anchored to original <question>
 - Transforms surface-level problems into deeper psychological insights
 - Makes the unconscious conscious
- The returned format must with the following fields:
  - action: The action you suggest to take as next step, action-answer or action-reflect
  - reflections: The reflections that you are reflecting on
    - question: The question that you are reflecting on
    - gaps_questions: You have found some sub-questions are important to the question, please list them here
    - think: You have tried to think out of the box or different angle, please list them here.
</action-reflect>""")

    sections.append(f"""
Based on the current context, you must choose one of the following actions:
<actions>
{(new_line + new_line).join(action_sections)}
</actions>""")

    # Add footer
    sections.append("Think step by step, choose the action, and respond in valid JSON format matching exact JSON schema of that action.")

    return remove_extra_line_breaks((new_line + new_line).join(sections))