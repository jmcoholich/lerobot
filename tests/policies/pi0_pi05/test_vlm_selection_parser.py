"""Exercise selection parsing without contacting a VLM or robot."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'vlm_client', Path(__file__).resolve().parents[3] / 'src/lerobot/policies/pi05/vlm_client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestSelectionParser(unittest.TestCase):
    def setUp(self):
        self.client = module.VLMClient.__new__(module.VLMClient)

    def test_json_formatting(self):
        for response in (
            '{"reasoning": "best", "chosen_trajectory": "blue"}',
            '{\n"chosen_trajectory": "blue",\n"reasoning": "best"\n}\n\n',
            '```json\n{"chosen_trajectory": "blue"}\n```',
            'My selection:\n{"chosen_trajectory": "blue"}\nDone.',
        ):
            with self.subTest(response=response):
                self.assertEqual(self.client._extract_chosen_color(response), 'blue')
        self.assertEqual(self.client._extract_chosen_color('{"chosen_trajectory": " up "}'), 'up')

    def test_invalid_or_ambiguous_selection(self):
        for response in (None, '', 'blue', '{}', '{"chosen_trajectory": null}',
                         '{"chosen_trajectory": ""}', '{"chosen_trajectory": "blue"',
                         '{"chosen_trajectory": "blue"}{"chosen_trajectory": "red"}'):
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    self.client._extract_chosen_color(response)
