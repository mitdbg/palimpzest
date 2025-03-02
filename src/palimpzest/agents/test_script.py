import unittest
import json

from palimpzest.agents.debugger_agent import DebuggerAgent 
import palimpzest.agents.utils as utils

class TestCodeExtractor(unittest.TestCase):
    def setUp(self):
        with open('test_strings.json', 'r') as file:
            self.test_strings = json.load(file)

        self.agent = DebuggerAgent()
        self.agent.relevant_issue_code = self.test_strings['relevant_issue_code']

    def test_get_classes_and_methods(self):
        result = self.agent.get_classes_and_methods("metadata.py")
        expected = '{\n    "MergeConflictError": [],\n    "MergeConflictWarning": [],\n    "MergeStrategyMeta": ["__new__"],\n    "MergeStrategy": ["merge"],\n    "MergePlus": ["merge"],\n    "MergeNpConcatenate": ["merge"],\n    "MetaData": ["__init__", "__get__", "__set__"],\n    "MetaAttribute": ["__init__", "__get__", "__set__", "__delete__", "__set_name__", "__repr__"]\n}'
        self.assertEqual(result, expected)

    def test_add_line_numbers(self):
        result = utils.add_line_numbers(self.test_strings['test_add_line_numbers'])
        expected = ''
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
