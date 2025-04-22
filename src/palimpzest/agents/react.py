import dspy
import json
from typing import Any
import tiktoken 

class ReAct(dspy.ReAct):
	"""
	A wrapper around dspy's ReAct module for custom functionality (e.g. context management) 
	"""

	def __init__(self, signature, tools: list[dspy.Tool], max_iters=5):
		"""
		Initialize the ReAct module with a signature and a list of tools.
		"""
		super().__init__(signature=signature, tools=tools, max_iters=max_iters)
		self._total_input_tokens = 0
		self._total_output_tokens = 0
	
	def get_total_input_tokens(self):
		"""
		Returns the total number of input tokens used in the ReAct module.
		"""
		return self._total_input_tokens

	def get_total_output_tokens(self):
		"""
		Returns the total number of output tokens used in the ReAct module.
		"""
		return self._total_output_tokens 

	def forward(self, **input_args):
		def format(trajectory: dict[str, Any], last_iteration: bool):
			adapter = dspy.settings.adapter or dspy.ChatAdapter()
			trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
			return adapter.format_fields(trajectory_signature, trajectory, role="user")
        
		trajectory = {}
		for idx in range(self.max_iters):

			try: 
				pred = self.react(**input_args, trajectory=format(trajectory, last_iteration=(idx == self.max_iters - 1)))
			except:
				# If ContextWindowErrorExceeded occurs, truncate the trajectory
				trajectory = self.truncate_trajectory(trajectory)
				pred = self.react(**input_args, trajectory=format(trajectory, last_iteration=(idx == self.max_iters - 1)))

			trajectory[f"thought_{idx}"] = pred.next_thought
			trajectory[f"tool_name_{idx}"] = pred.next_tool_name
			trajectory[f"tool_args_{idx}"] = pred.next_tool_args

			try:
				# DSPy bug fix 
				tool_args = json.loads(pred.next_tool_args) if isinstance(pred.next_tool_args, str) else pred.next_tool_args
				# Running tool function
				trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**tool_args)
			except Exception as e:
				trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"

			if pred.next_tool_name == "finish":
				break
		
		# Compute the total number of input and output tokens
		self.compute_past_n_total_tokens(n=len(trajectory))

		extract = self.extract(**input_args, trajectory=format(trajectory, last_iteration=False))
		return dspy.Prediction(trajectory=trajectory, **extract)

	def truncate_trajectory(self, trajectory: dict, width: int = 20000, token_limit: int = 120000):
		""" 
		Truncates the trajectory to contain at most (token_limit - width) number of tokens
		"""

		# Count tokens 
		total_tokens = self.count_trajectory_tokens(trajectory)
		encoding = tiktoken.encoding_for_model("gpt-4")  

		# Truncate tokens
		if token_limit - width >= 0:
			for step, content in trajectory.items():
				if total_tokens <= token_limit - width:
					break

				# Archive the step 
				trajectory[step] = "content archived..."

				# Recalculate total tokens
				total_tokens -= len(encoding.encode(content)) 

	
	def count_trajectory_tokens(trajectory: dict[str, str], model: str = "gpt-4") -> int:
		"""
		Counts the number of tokens in trajectory 
		"""
		encoding = tiktoken.encoding_for_model(model)
		total_tokens = sum(len(encoding.encode(key)) for key in trajectory.keys())
		return total_tokens

	def compute_past_n_total_tokens(self, n: int):
		"""
		Given a list of data items, returns the total input (prompt) and output (completion) tokens
		across the last n items.

		Args:
			data (list): List of dictionaries, each containing a 'usage' field.
			n (int): Number of most recent items to consider.

		Returns:
			dict: A dictionary with total 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
		"""
		total_prompt = 0
		total_completion = 0
		data = dspy.settings.lm.history

		# Get the last n items
		for entry in data[-n:]:
			usage = entry.get('usage', {})
			total_prompt += usage.get('prompt_tokens', 0)
			total_completion += usage.get('completion_tokens', 0)
		
		self._total_input_tokens = total_prompt
		self._total_output_tokens = total_completion