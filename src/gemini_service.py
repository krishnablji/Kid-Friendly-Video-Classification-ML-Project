"""
gemini_service.py
Integrates Google GenAI SDK (gemini-2.5-flash) to process uploaded videos
and automatically generate vast, detailed scene-by-scene summaries.
Includes mock/demo responses for offline testing without burning API credits.
"""

import os
import time
import tempfile


class GeminiVideoService:
    def __init__(self, api_key: str = None):
        """
        Initializes Gemini client. Reads api_key argument or GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Notice: Failed to initialize Google GenAI client: {e}")

    def summarize_video(self, video_file_path: str, model_name: str = "gemini-2.5-flash") -> str:
        """
        Uploads video to Gemini API, waits for processing, and generates vast scene summary.
        Prompt is applied automatically without requiring manual user input.
        """
        if not self.client:
            raise ValueError("GEMINI_API_KEY not configured. Please provide an API key or use Demo Mode.")

        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")

        from google import genai

        print(f"Uploading {video_file_path} to Gemini File API...")
        uploaded_file = self.client.files.upload(file=video_file_path)

        # Wait for video file to be in ACTIVE state
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini video processing failed: {uploaded_file.error.message}")

        # Automated system prompt - zero manual prompting needed
        prompt = (
            "Please provide an objective, comprehensive, sentence-by-sentence description and summary "
            "of all visuals, actions, people, objects, audio cues, dialogues, and themes present in this video clip."
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, prompt]
            )
            summary_text = response.text
        finally:
            # Clean up uploaded video from Gemini servers
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

        return summary_text


# Built-in multi-sentence benchmark summaries for instant testing & demonstration
PRESET_DEMO_TEXTS = {
    "Cartoon Alphabet & Animals (Completely Safe)": (
        "The video displays a brightly colored 2D animated playground on a sunny morning. "
        "A friendly cartoon elephant and rabbit hop onto the green grass and wave cheerfully at the audience. "
        "Upbeat acoustic music plays while animated alphabet letters float gently down from fluffy clouds. "
        "The animal characters sing along to teach children the letter sounds. "
        "Children smile and applaud as the animals take a bow and the video fades out smoothly."
    ),
    "Action Movie Clip with Violence (Triggers Early-Exit)": (
        "The video opens with a cinematic wide aerial view of a busy downtown street at sunset. "
        "Cars drive smoothly down the boulevard while ambient urban traffic sounds play. "
        "A sleek silver sedan turns sharply around the corner and parks near an office building. "
        "The video captures a violent street fight where a person is stabbed with a knife and bleeding heavily on the ground. "
        "A large crowd of pedestrians gathers around the scene taking photos with their phones. "
        "Emergency sirens echo in the distance as police vehicles arrive at the intersection."
    ),
    "Science Experiment for Kids (Completely Safe)": (
        "The video showcases a classroom kitchen counter set up with colorful measuring cups and safety goggles. "
        "An instructor in a lab coat explains how baking soda and red food coloring react together. "
        "Two elementary school students measure vinegar into a clear plastic flask with big smiles. "
        "The mixture bubbles up gently like a mini toy volcano causing the children to giggle with excitement. "
        "The teacher explains the chemical reaction in simple, easy-to-understand words for kids."
    ),
    "Haunted Corridor / Threat (Triggers Early-Exit)": (
        "The video shows an exterior shot of an old wooden cabin surrounded by tall pine trees in the fog. "
        "Gentle wind rustles through the leaves as the camera tracks toward the front door. "
        "The video shows a man holding a handgun pointing it at another person's chest and firing multiple gunshots. "
        "Dark shadows flicker across the hallway while footsteps echo on the floorboards. "
        "The scene fades into a dark closing title card."
    )
}
