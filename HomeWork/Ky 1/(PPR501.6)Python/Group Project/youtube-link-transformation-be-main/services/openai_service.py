import os
import openai
import json

from services.exceptions import OpenAIServiceException

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text_with_openai(
    text: str,
    desired_word_count: int,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Summarize the text using OpenAI's GPT model in ~desired_word_count words.
    Uses chunking if the transcript is very large, then merges partial summaries.
    """
    if not openai.api_key:
        raise OpenAIServiceException("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    # Approx. 1 token ~ 4 chars => ~3000 tokens * 4 = 12000 chars
    chunk_size = 12000
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end

    partial_summaries = []

    # Summarize each chunk
    for chunk in chunks:
        prompt = (
            "You are a helpful assistant. Please provide a concise summary "
            f"of the following text in about {desired_word_count} words:\n\n{chunk}\n\n"
            "Summary:"
        )
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            chunk_summary = response.choices[0].message["content"].strip()
            partial_summaries.append(chunk_summary)
        except Exception as exc:
            raise OpenAIServiceException(f"Error summarizing text chunk: {str(exc)}")

    # If we have multiple partial summaries, combine them
    if len(partial_summaries) > 1:
        combined_text = "\n".join(partial_summaries)
        prompt = (
            "You are a helpful assistant. I have multiple partial summaries of a larger text. "
            f"Please combine them into one coherent summary of about {desired_word_count} words:\n\n"
            f"{combined_text}\n\n"
            "Final summary:"
        )
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            final_summary = response.choices[0].message["content"].strip()
            return final_summary
        except Exception as exc:
            raise OpenAIServiceException(f"Error combining partial summaries: {str(exc)}")
    else:
        # Only one chunk
        return partial_summaries[0]


def create_outline_with_openai(transcript: str, model="gpt-3.5-turbo") -> str:
    """
    Uses GPT to transform a raw transcript into a bullet-point outline.
    """
    if not openai.api_key:
        raise OpenAIServiceException("OpenAI API key not found.")

    prompt = f"""
You are an expert at organizing unstructured text into a hierarchical outline.
Read the following transcript and create a bullet-point outline.
Use indentation (with spaces or dashes) to show sub-topics.

Example:
- Topic
  - Subtopic
    - Sub-subtopic

Return ONLY the bullet-point outline (no extra commentary). 

Transcript:
{transcript}
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        outline = response.choices[0].message["content"].strip()
        return outline
    except Exception as exc:
        raise OpenAIServiceException(f"Error creating outline: {str(exc)}")


def generate_mind_map_with_openai(outline: str, model="gpt-3.5-turbo") -> dict:
    """
    Converts a bullet-point outline into mind map JSON data with `nodes` and `links`,
    enforcing a root node with a 5-10 word summary and ensuring no standalone nodes.
    """
    global raw_output
    if not openai.api_key:
        raise OpenAIServiceException("OpenAI API key not found.")

    # Prompt to create a root summary node and ensure that all nodes
    # link together under that root.
    prompt = f"""
You are an expert in creating mind maps as graph data.

I will give you a text outline with multiple levels of bullet points, for example:
- Introduction
  - Purpose
    - Overview

**Your tasks**:
1. Analyze the entire outline and create a single root node that is a short 5-10 word summary of the entire transcript.
   - Use "Root" or "Summary" or something similar only if needed, but the text itself must be 5-10 words.
2. Convert each bullet (each line) of the given outline into a node: the bullet text becomes "id".
3. Determine the "group" based on indentation level. Group 0 for the root node, Group 1 for its direct children, Group 2 for their children, etc.
4. Create parent->child links, each with "value": 1. No node can be stand-alone; all must ultimately link back to the root node.
5. Return ONLY valid JSON with this structure (no markdown, no extra text):
{{
  "nodes": [
    {{"id": "string", "group": 0}},
    ...
  ],
  "links": [
    {{"source": "ParentID", "target": "ChildID", "value": 1}},
    ...
  ]
}}

Outline to convert:
{outline}
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_output = response.choices[0].message["content"].strip()
        mind_map_data = json.loads(raw_output)  # Validate JSON
        return mind_map_data
    except json.JSONDecodeError as e:
        raise OpenAIServiceException(
            f"The model output was not valid JSON. Raw output: {raw_output} Error: {e}"
        )
    except Exception as exc:
        raise OpenAIServiceException(f"Error generating mind map: {str(exc)}")
