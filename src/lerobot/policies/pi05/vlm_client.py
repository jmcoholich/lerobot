import base64
import json
import re
import requests
from io import BytesIO
from typing import List, Optional, Tuple
import torch
from PIL import Image
import random
import logging
log = logging.getLogger(__name__)


class VLMClient:
    """
    A simple VLM client to interact with OpenAI-compatible VLM servers.
    Sends images with trajectory visualizations to the VLM and gets trajectory selection.
    """

    def __init__(self, server_url: str, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        """
        Initialize the VLM Client.

        Args:
            server_url: The base URL of the OpenAI-compatible VLM server.
            model_name: The model name identifier.
        """
        self.server_url = server_url
        self.endpoint = f"{server_url}/v1/chat/completions"
        self.model_name = model_name

        # Store intermediate text responses
        self.last_text_responses = []

        # Health check
        health_url = server_url.rstrip('/') + "/health"
        try:
            health_response = requests.get(health_url, timeout=5)
            health_response.raise_for_status()

            print(f"Successfully connected to VLM server at {server_url}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to VLM server at {health_url}: {e}")
            raise RuntimeError

    def _pil_to_base64(self, image: Image.Image) -> str:
        """Converts a PIL Image to a base64 encoded string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _extract_chosen_color(self, text_output):
        last_line = text_output.split('\n')[-2]
        if not "chosen_trajectory" in last_line:
            raise RuntimeError("VLM output invalid")
        color = last_line.split(":")[-1]
        color = color.replace('"', '').strip()
        return color



    def select_trajectories(
        self,
        annotated_image: Image.Image,
        prompt_text: str,
        num_trajectories: int,
        max_new_tokens: int = 1024,
        timeout: int = 60
    ) -> tuple[Tuple[int, int], str]:
        """
        Send an annotated image to the VLM and get trajectory selections for both arms.

        Returns:
            tuple: ((idx_left, idx_right), text_response)
        """
        user_content = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._pil_to_base64(annotated_image)}"
                }
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert AI controller for a tabletop manipulation task."},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]

            chosen_color = self._extract_chosen_color(generated_text)
            self.last_text_responses = [generated_text]

            return chosen_color, generated_text

        except Exception as e:
            print(f"VLM Request failed: {e}. Is the model type correct?")
            return (0, 0), str(e)

    def _extract_chosen_primitives(self, response_text: str) -> dict:
        """
        Extracts 'chosen_primitive_left', 'chosen_primitive_right',
        'gripper_state_left', and 'gripper_state_right' from VLM response.
        Handles standard JSON and Markdown formatting.

        Args:
            response_text (str): The raw string returned by the VLM.

        Returns:
            dict: {'chosen_primitive_left': str, 'chosen_primitive_right': str,
                   'gripper_state_left': str, 'gripper_state_right': str}
                Values are None if extraction fails.
        """
        # Default return structure
        extracted_data: dict[str, str] = {
            "chosen_primitive_left": "None",
            "chosen_primitive_right": "None",
            "gripper_state_left": "None",
            "gripper_state_right": "None"
        }

        # 1. Clean up Markdown code blocks (e.g., ```json ... ```)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()

        # 2. Attempt strict JSON parsing
        try:
            data = json.loads(clean_text)
            extracted_data["chosen_primitive_left"] = data.get("chosen_primitive_left")
            extracted_data["chosen_primitive_right"] = data.get("chosen_primitive_right")
            extracted_data["gripper_state_left"] = data.get("gripper_state_left")
            extracted_data["gripper_state_right"] = data.get("gripper_state_right")
            return extracted_data
        except json.JSONDecodeError:
            log.warning(f"JSON parse failed for VLM response. Attempting Regex fallback. Response: {response_text[:50]}...")

        # 3. Fallback: Regex extraction
        # This handles cases where the VLM adds extra text outside the JSON block

        # Extract chosen_primitive_left
        traj_pattern_left = r'"chosen_primitive_left"\s*:\s*"([^"]+)"'
        traj_match_left = re.search(traj_pattern_left, response_text)
        if traj_match_left:
            extracted_data["chosen_primitive_left"] = traj_match_left.group(1)

        # Extract chosen_primitive_right
        traj_pattern_right = r'"chosen_primitive_right"\s*:\s*"([^"]+)"'
        traj_match_right = re.search(traj_pattern_right, response_text)
        if traj_match_right:
            extracted_data["chosen_primitive_right"] = traj_match_right.group(1)

        # Extract gripper_state_left
        state_pattern_left = r'"gripper_state_left"\s*:\s*"([^"]+)"'
        state_match_left = re.search(state_pattern_left, response_text)
        if state_match_left:
            extracted_data["gripper_state_left"] = state_match_left.group(1)

        # Extract gripper_state_right
        state_pattern_right = r'"gripper_state_right"\s*:\s*"([^"]+)"'
        state_match_right = re.search(state_pattern_right, response_text)
        if state_match_right:
            extracted_data["gripper_state_right"] = state_match_right.group(1)

        return extracted_data

    def select_primitives(
        self,
        annotated_image: Image.Image,
        prompt_text: str,
        primitive_names: List[str],
        max_new_tokens: int = 1024,
        timeout: int = 60
    ) -> tuple[Tuple[int, int], str]:
        """
        Send an annotated image to the VLM and get trajectory selections for both arms.

        Returns:
            tuple: (dict[str, int], text_response)
        """
        user_content = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._pil_to_base64(annotated_image)}"
                }
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert robot policy advisor specializing in dual-arm coordination."},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]

            extracted_data = self._extract_chosen_primitives(generated_text)
            for key, value in extracted_data.items():
                if value == "None" or value is None:
                    extracted_data[key] = 0
                else:
                    if 'chosen_primitive' in key:
                        try:
                            extracted_data[key] = primitive_names.index(value)
                        except ValueError:
                            extracted_data[key] = 0
                    else: # Gripper_state
                        if value.lower() == "open":
                            extracted_data[key] = 1.0
                        else:
                            extracted_data[key] = 0.0

            return extracted_data, generated_text

        except Exception as e:
            print(f"VLM Request failed: {e}. Defaulting to (0, 0).")
            return {"chosen_primitive_left": 0, "chosen_primitive_right": 0, "gripper_state_left": 0, "gripper_state_right": 0}, str(e)

    def get_last_text_responses(self) -> List[str]:
        """Get the text responses from the last VLM call."""
        return self.last_text_responses

if __name__ == "__main__":
    # ssh -N -f -L localhost:60721:localhost:60721 -J jcoholich3@sky1.cc.gatech.edu jcoholich3@optimistprime.cc.gatech.edu
    client = VLMClient("http://127.0.0.1:60721")