"""Exercise VLM requests, metrics, plots, and parsing without a server or robot."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageFont

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


class TestVLMRequests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4, 3))
        self.reply = {
            "chosen_trajectory": "blue",
            "chosen_primitive_left": "left",
            "chosen_primitive_right": "right",
            "gripper_state_left": "open",
            "gripper_state_right": "closed",
        }

    def check_request(self, client, method_name, post):
        text = module.json.dumps(self.reply)
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        }
        choices = 2 if method_name == "select_trajectories" else ["left", "right"]
        with patch.object(module.time, "perf_counter", side_effect=[10.0, 12.5]):
            selection, response = getattr(client, method_name)(
                self.image, "Choose a trajectory", choices, max_new_tokens=512, timeout=17
            )
        self.assertEqual(client.last_response_metrics["latency_s"], 2.5)
        self.assertEqual(client.last_response_metrics["usage"],
                         {"prompt_tokens": 1000, "completion_tokens": 200})
        self.assertEqual(response, text)
        if method_name == "select_trajectories":
            self.assertEqual(selection, "blue")
        else:
            self.assertEqual(selection, {
                "chosen_primitive_left": 0, "chosen_primitive_right": 1,
                "gripper_state_left": 1.0, "gripper_state_right": 0.0,
            })
        self.assertEqual(post.call_args.args, (client.endpoint,))
        self.assertEqual(post.call_args.kwargs["timeout"], 17)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], client.model_name)
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Choose a trajectory"})
        image_url = content[1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        with Image.open(module.BytesIO(module.base64.b64decode(image_url.split(",", 1)[1]))) as image:
            self.assertEqual(image.size, self.image.size)
        return payload, post.call_args.kwargs["headers"]

    def test_openai_requests(self):
        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "get") as get, \
                patch.object(module.requests, "post") as post:
            for url in ("https://api.openai.com", "https://api.openai.com/", "https://api.openai.com/v1/"):
                for method in ("select_trajectories", "select_primitives"):
                    with self.subTest(url=url, method=method):
                        client = module.VLMClient(url, "gpt-4.1")
                        self.assertEqual(client.endpoint, "https://api.openai.com/v1/chat/completions")
                        payload, headers = self.check_request(client, method, post)
                        self.assertEqual(headers["Authorization"], "Bearer test-key")
                        self.assertEqual(payload["max_completion_tokens"], 512)
                        for key in ("max_tokens", "temperature", "chat_template_kwargs"):
                            self.assertNotIn(key, payload)
            get.assert_not_called()

    def test_local_requests_preserve_options_without_sending_openai_key(self):
        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "get") as get, \
                patch.object(module.requests, "post") as post:
            for model in ("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.6", "Qwen/Qwen3.8", "Qwen/Qwen3-A3B"):
                for method in ("select_trajectories", "select_primitives"):
                    with self.subTest(model=model, method=method):
                        client = module.VLMClient("http://127.0.0.1:51995/", model)
                        get.assert_called_with("http://127.0.0.1:51995/health", timeout=5)
                        self.assertEqual(client.endpoint, "http://127.0.0.1:51995/v1/chat/completions")
                        payload, headers = self.check_request(client, method, post)
                        self.assertNotIn("Authorization", headers)
                        self.assertNotIn("max_completion_tokens", payload)
                        self.assertEqual(payload["max_tokens"], 512)
                        self.assertEqual(payload["temperature"], 0.0)
                        if model == "Qwen/Qwen3-VL-8B-Instruct":
                            self.assertNotIn("chat_template_kwargs", payload)
                        else:
                            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_openai_requires_key_before_any_network_call(self):
        with patch.dict(module.os.environ, {}, clear=True), \
                patch.object(module.requests, "get") as get, \
                patch.object(module.requests, "post") as post:
            for key in (None, "", "   "):
                with self.subTest(key=key):
                    if key is not None:
                        module.os.environ["OPENAI_API_KEY"] = key
                    with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY.*policy server"):
                        module.VLMClient("https://api.openai.com", "gpt-4.1")
            get.assert_not_called()
            post.assert_not_called()

    def test_openai_key_requires_exact_host_and_https(self):
        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "get"):
            client = module.VLMClient("https://api.openai.com.example.org", "gpt-4.1")
            self.assertNotIn("Authorization", client.headers)
            with self.assertRaisesRegex(ValueError, "https"):
                module.VLMClient("http://api.openai.com", "gpt-4.1")

    def test_local_health_failure_still_raises(self):
        with patch.object(module.requests, "get", side_effect=module.requests.ConnectionError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "VLM health check failed"):
                module.VLMClient("http://127.0.0.1:51995")

    def test_failed_request_clears_previous_metrics(self):
        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "post", side_effect=module.requests.Timeout("timed out")):
            client = module.VLMClient("https://api.openai.com", "gpt-4.1")
            client.last_response_metrics = {"usage": {"prompt_tokens": 1000}}
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                client.select_trajectories(self.image, "Choose", 2)
            self.assertIsNone(client.last_response_metrics)

    def test_parallel_requests_overlap_and_keep_metrics_separate(self):
        barrier = Barrier(2, timeout=5)

        def respond(endpoint, *, json, headers, timeout):
            # Sequential dispatch cannot pass this barrier.
            barrier.wait()
            prompt = json["messages"][1]["content"][0]["text"]
            count = 123 if prompt == "pivot" else 456
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": module.json.dumps({"chosen_trajectory": prompt})}}],
                "usage": {"prompt_tokens": count, "completion_tokens": count // 3},
            }
            return response

        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "post", side_effect=respond):
            client = module.VLMClient("https://api.openai.com", "gpt-4.1")
            results = client.select_trajectories_batch([
                (self.image, "pivot", 2), (self.image, "primitive", 7),
            ])
        self.assertEqual([r[0] for r in results], ["pivot", "primitive"])
        self.assertEqual([r[2]["usage"]["prompt_tokens"] for r in results], [123, 456])
        self.assertEqual([r[2]["usage"]["completion_tokens"] for r in results], [41, 152])
        self.assertIsNot(results[0][2], results[1][2])
        total = results[0][2]["batch_latency_s"]
        self.assertEqual(results[1][2]["batch_latency_s"], total)
        for _, _, metrics in results:
            self.assertGreater(metrics["latency_s"], 0)
            self.assertGreaterEqual(total, metrics["latency_s"])
        self.assertIsNone(client.last_response_metrics)
        self.assertEqual(client.get_last_text_responses(), [r[1] for r in results])

    def test_parallel_request_failure_propagates(self):
        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "post", side_effect=module.requests.Timeout("timed out")):
            client = module.VLMClient("https://api.openai.com", "gpt-4.1")
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                client.select_trajectories_batch([(self.image, "pivot", 2), (self.image, "primitive", 7)])
            self.assertIsNone(client.last_response_metrics)
            self.assertEqual(client.get_last_text_responses(), [])

    def test_serial_requests_finish_in_order_and_keep_metrics_separate(self):
        events = []

        def respond(endpoint, *, json, headers, timeout):
            prompt = json["messages"][1]["content"][0]["text"]
            events.append((prompt, "start"))
            response = Mock()

            def body():
                events.append((prompt, "finish"))
                return {
                    "choices": [{"message": {"content": module.json.dumps({"chosen_trajectory": prompt})}}],
                    "usage": {"prompt_tokens": 100 if prompt == "pivot" else 200},
                }

            response.json.side_effect = body
            return response

        with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(module.requests, "post", side_effect=respond):
            client = module.VLMClient("https://api.openai.com", "gpt-4.1")
            results = client.select_trajectories_batch(
                [(self.image, "pivot", 2), (self.image, "primitive", 7)], mode="serial",
            )
        self.assertEqual(events, [("pivot", "start"), ("pivot", "finish"),
                                  ("primitive", "start"), ("primitive", "finish")])
        self.assertEqual([r[2]["usage"]["prompt_tokens"] for r in results], [100, 200])
        self.assertTrue(all(r[2]["request_mode"] == "serial" for r in results))
        total = results[0][2]["batch_latency_s"]
        self.assertEqual(results[1][2]["batch_latency_s"], total)
        self.assertGreaterEqual(total, sum(r[2]["latency_s"] for r in results))


class TestResponsePlot(unittest.TestCase):
    def test_metrics_are_rendered_on_both_response_plots(self):
        # Load the rendering function without importing the policy/model/hardware.
        source = Path(module.__file__).with_name("modelling_pi05_taco.py")
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "save_VLM_io")
        namespace = dict(Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont, Path=Path)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            for suffix, usage in (("primitive", {"prompt_tokens": 1000, "completion_tokens": 200,
                                                  "completion_tokens_details": {"reasoning_tokens": 150}}),
                                  ("pivot", {}),
                                  ("pivot", {"prompt_tokens": 10, "completion_tokens": 0,
                                             "completion_tokens_details": {"reasoning_tokens": 0}}),
                                  ("primitive", {"completion_tokens_details": None})):
                with patch.object(ImageDraw.ImageDraw, "text", autospec=True) as draw_text:
                    path = namespace["save_VLM_io"](
                        Image.new("RGB", (360, 360)), "Selected blue", 1, output_dir=directory,
                        suffix=suffix, response_metrics={"model": "gpt-6-astra", "latency_s": 2.5,
                                                        "batch_latency_s": 3.0,
                                                        "request_mode": "serial" if suffix == "primitive" else "parallel",
                                                        "usage": usage},
                    )
                rendered = " ".join(call.args[2] for call in draw_text.call_args_list)
                self.assertIn("Full response latency: 2.50 s", rendered)
                self.assertIn(f"Both {'serial' if suffix == 'primitive' else 'parallel'} calls latency: 3.00 s", rendered)
                self.assertIn(f"Input tokens: {usage.get('prompt_tokens', 'unavailable')}", rendered)
                self.assertIn(f"Output tokens: {usage.get('completion_tokens', 'unavailable')}", rendered)
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "unavailable")
                self.assertIn(f"Reasoning tokens (included in output): {reasoning}", rendered)
                self.assertNotIn("cost", rendered.lower())
                self.assertIn("Selected blue", rendered)
                self.assertTrue(path.name.endswith(f"_{suffix}.png"))
                with Image.open(path) as saved:
                    self.assertGreater(saved.height, 360)
