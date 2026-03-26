import os
import gradio as gr
import requests
import pandas as pd
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# --- Tools ---
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage and returns its text content.

    Args:
        url: The URL of the webpage to visit.
    """
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        # Simple HTML to text - strip tags
        from html.parser import HTMLParser
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self.skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self.skip = False
            def handle_data(self, data):
                if not self.skip:
                    self.result.append(data)
        extractor = HTMLTextExtractor()
        extractor.feed(response.text)
        text = " ".join(extractor.result).strip()
        # Limit length
        return text[:10000] if len(text) > 10000 else text
    except Exception as e:
        return f"Error visiting {url}: {e}"


@tool
def read_file_from_api(task_id: str) -> str:
    """Downloads and reads a file associated with a GAIA task.

    Args:
        task_id: The task ID to download the file for.
    """
    try:
        api_url = DEFAULT_API_URL
        response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text" in content_type or "json" in content_type or "csv" in content_type:
            return response.text[:10000]
        else:
            return f"Binary file received ({content_type}), {len(response.content)} bytes. Cannot display as text."
    except Exception as e:
        return f"Error downloading file for task {task_id}: {e}"


# --- Agent Definition ---
class GaiaAgent:
    def __init__(self):
        model = HfApiModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=os.getenv("HF_TOKEN"),
        )
        self.agent = CodeAgent(
            tools=[DuckDuckGoSearchTool(), visit_webpage, read_file_from_api],
            model=model,
            max_steps=10,
            verbosity_level=1,
        )
        print("GaiaAgent initialized with smolagents CodeAgent.")

    def __call__(self, question: str, task_id: str = None) -> str:
        print(f"Agent received question (first 80 chars): {question[:80]}...")
        prompt = f"""Answer the following question precisely and concisely.
If the question requires web search, use the search tool.
If the question references a file, use read_file_from_api with task_id "{task_id}".
Give ONLY the final answer with no extra explanation.

Question: {question}"""
        try:
            result = self.agent.run(prompt)
            answer = str(result).strip()
            print(f"Agent answer: {answer}")
            return answer
        except Exception as e:
            print(f"Agent error: {e}")
            return f"Error: {e}"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches all questions, runs the agent, submits answers, and displays results."""
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # 3. Run Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text, task_id=task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        return "Agent did not produce any answers.", pd.DataFrame(results_log)

    # 4. Submit
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers...")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', '')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        1. Log in with your Hugging Face account.
        2. Click the button to run the agent on all GAIA questions and submit.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Answers", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("Launching GAIA Agent...")
    demo.launch(debug=True, share=False)
