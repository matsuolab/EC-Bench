"""Centralized English prompt templates used across the project.

Each entry captures the current understanding of how a downstream caller
constructs its request. Dynamic portions (lists of questions, frame data, etc.)
should be injected by the caller via the documented placeholders.
"""

from textwrap import dedent


PROMPTS = {
    "generate_sampling_openQA": {
        "description": (
            "Sampling frames + transcription → free-form answer generation "
            "prompt. Inject the formatted sections for queries, frames, and "
            "transcription into the placeholders."
        ),
        "template": dedent(
            """
            You are a dataset annotator specializing in counting tasks for long-form videos (30+ minutes).
            Please analyze the given sampled video frames and audio transcription in detail, generate accurate answers for all the following queries.

            {queries_section}

            {frames_section}

            {transcription_section}

            ## Answer Rules

            ### 1. Counting Type
            **Definition**: Questions that require numerical answers such as "frequency," "number of people," or "percentage" of similar entities or events.
            **Recording Rules**:
            - Basic format: Numbers only
            - Keep it concise: Record only numbers without converting to sentences
            - When multiple numbers are needed: Separate with commas (e.g., "2, 5, 7")
            - **5-second rule**: If the same subject reappears within 5 seconds, count it as one occurrence

            ### 2. Enumeration Type
            **Definition**: Questions that require listing multiple entities or procedures as items.
            **Recording Rules**:
            - Basic format: List items separated by commas
            - When order is important (causal/procedural enumeration): Connect processes with "→"
            - No unnecessary explanations: Only item names or procedure names
            - Omit owner names: For simple object enumeration, answer with nouns only (e.g., "A's car" → "car")
            **Recording Examples**:
            - (Simple enumeration) "bat, glove, ball"
            - (Person enumeration) "Player A, Player B, Player C"
            - (Process enumeration) "forging→polishing→decoration→completion"
            - (Process enumeration) "facial recognition→amount confirmation→approval→payment completion"

            ## Analysis Instructions
            1. Carefully examine all provided video frames in chronological order
            2. Use the audio transcription to understand the context and dialogue
            3. Combine visual and audio information to generate accurate answers
            4. Pay attention to temporal relationships between frames when counting or enumerating

            ## Output Format
            Please output answers for all queries in the following JSON format:

            ```json
            {{
              "results": [
                {{
                  "query_id": 1,
                  "answer": "accurate answer"
                }},
                {{
                  "query_id": 2,
                  "answer": "accurate answer"
                }}
              ]
            }}
            ```

            **Important Notes**:
            - **Type Determination**: Please select the appropriate answer format according to each query type
              - Counting: Answer concisely with numbers only
              - Enumeration: List items/procedures separated by commas or "→"
            - Each answer must strictly follow the above rules and be generated accurately based on the sampled frames and audio content
            - Use both visual information from frames and contextual information from audio transcription
            """
        ).strip(),
        "placeholders": ["queries_section", "frames_section", "transcription_section"],
    },
    "generate_sampling_openQA_with_clips": {
        "description": (
            "Adds evidence clip collection to the sampling openQA prompt. Callers must "
            "inject queries, frame summaries, and transcription blocks into the placeholders."
        ),
        "template": dedent(
            """
            You are a dataset annotator specializing in counting tasks for long-form videos (30+ minutes).
            Please analyze the given sampled video frames and audio transcription in detail, generate accurate answers for all the following queries, and record the evidence clips that support each answer.

            {queries_section}

            {frames_section}

            {transcription_section}

            ## Answer Rules

            ### 1. Counting Type
            **Definition**: Questions that require numerical answers such as "frequency," "number of people," or "percentage" of similar entities or events.
            **Recording Rules**:
            - Basic format: Numbers only
            - Keep it concise: Record only numbers without converting to sentences
            - When multiple numbers are needed: Separate with commas (e.g., "2, 5, 7")
            - **5-second rule**: If the same subject reappears within 5 seconds, count it as one occurrence

            ### 2. Enumeration Type
            **Definition**: Questions that require listing multiple entities or procedures as items.
            **Recording Rules**:
            - Basic format: List items separated by commas
            - When order is important (causal/procedural enumeration): Connect processes with "→"
            - No unnecessary explanations: Only item names or procedure names
            - Omit owner names: For simple object enumeration, answer with nouns only (e.g., "A's car" → "car")
            **Recording Examples**:
            - (Simple enumeration) "bat, glove, ball"
            - (Person enumeration) "Player A, Player B, Player C"
            - (Process enumeration) "forging→polishing→decoration→completion"
            - (Process enumeration) "facial recognition→amount confirmation→approval→payment completion"

            ## Analysis Instructions
            1. Carefully examine all provided video frames in chronological order
            2. Use the audio transcription to understand the context and dialogue
            3. Combine visual and audio information to generate accurate answers
            4. Pay attention to temporal relationships between frames when counting or enumerating
            5. For each answer, identify the precise video clips that support your conclusion

            ## Output Format
            Please output answers for all queries in the following JSON format:

            ```json
            {{
              "results": [
                {{
                  "query_id": 1,
                  "answer": "accurate answer",
                  "clips": [
                    ["00:04:12", "00:07:23"],
                    ["00:12:12", "00:12:56"]
                  ]
                }}
              ]
            }}
            ```

            **Important Notes**:
            - **Type Determination**: Please select the appropriate answer format according to each query type
              - Counting: Answer concisely with numbers only
              - Enumeration: List items/procedures separated by commas or "→"
            - `clips` must list every video interval that served as evidence for the answer, using `[start_timestamp, end_timestamp]` pairs. Include all intervals needed to justify the response.
            - Each answer must strictly follow the above rules and be generated accurately based on the sampled frames and audio content
            - Use both visual information from frames and contextual information from audio transcription
            """
        ).strip(),
        "placeholders": ["queries_section", "frames_section", "transcription_section"],
    },
    "generate_fullvideo_openQA_with_clips": {
        "description": (
            "Full-video input variant of generate_sampling_openQA_with_clips. "
            "The entire video is passed to the model directly (no frame sampling). "
            "Inject queries, video info, and transcription into the placeholders."
        ),
        "template": dedent(
            """
            You are a dataset annotator specializing in counting tasks for long-form videos (30+ minutes).
            Please analyze the given full video and audio transcription in detail, generate accurate answers for all the following queries, and record the evidence clips that support each answer.

            {queries_section}

            {video_section}

            {transcription_section}

            ## Answer Rules

            ### 1. Counting Type
            **Definition**: Questions that require numerical answers such as "frequency," "number of people," or "percentage" of similar entities or events.
            **Recording Rules**:
            - Basic format: Numbers only
            - Keep it concise: Record only numbers without converting to sentences
            - When multiple numbers are needed: Separate with commas (e.g., "2, 5, 7")
            - **5-second rule**: If the same subject reappears within 5 seconds, count it as one occurrence

            ### 2. Enumeration Type
            **Definition**: Questions that require listing multiple entities or procedures as items.
            **Recording Rules**:
            - Basic format: List items separated by commas
            - When order is important (causal/procedural enumeration): Connect processes with "→"
            - No unnecessary explanations: Only item names or procedure names
            - Omit owner names: For simple object enumeration, answer with nouns only (e.g., "A's car" → "car")
            **Recording Examples**:
            - (Simple enumeration) "bat, glove, ball"
            - (Person enumeration) "Player A, Player B, Player C"
            - (Process enumeration) "forging→polishing→decoration→completion"
            - (Process enumeration) "facial recognition→amount confirmation→approval→payment completion"

            ## Analysis Instructions
            1. Watch the full video carefully from beginning to end
            2. Use the audio transcription to understand the context and dialogue
            3. Combine visual and audio information to generate accurate answers
            4. Pay attention to temporal relationships when counting or enumerating
            5. For each answer, identify the precise video clips that support your conclusion

            ## Output Format
            Please output answers for all queries in the following JSON format:

            ```json
            {{
              "results": [
                {{
                  "query_id": 1,
                  "answer": "accurate answer",
                  "clips": [
                    ["00:04:12", "00:07:23"],
                    ["00:12:12", "00:12:56"]
                  ]
                }}
              ]
            }}
            ```

            **Important Notes**:
            - **Type Determination**: Please select the appropriate answer format according to each query type
              - Counting: Answer concisely with numbers only
              - Enumeration: List items/procedures separated by commas or "→"
            - `clips` must list every video interval that served as evidence for the answer, using `[start_timestamp, end_timestamp]` pairs. Include all intervals needed to justify the response.
            - Each answer must strictly follow the above rules and be generated accurately based on the full video and audio content
            - Use both visual information from the video and contextual information from audio transcription
            """
        ).strip(),
        "placeholders": ["queries_section", "video_section", "transcription_section"],
    },
    "generate_e_to_c_openQA": {
        "description": (
            "Forces a Counting task to enumerate first, then derive the count. Intended for E→C sequential answering."
        ),
        "template": dedent(
            """
            You are an expert annotator who analyzes long-form videos (30+ minutes) with high precision for Counting tasks.
            Using the information below, first enumerate every relevant instance, then derive the final count from that list.

            {query_section}

            {frames_section}

            {transcription_section}

            ## Instructions
            1. The given query is a Counting task. You must follow a "enumerate first, then count" workflow.
            2. During enumeration, list each distinct instance as bullet points, adding any necessary identifiers (appearance, action, timestamp, etc.) to distinguish them.
            3. Once enumeration is complete, deduplicate the list if needed and calculate the precise total.
            4. Combine evidence from both sampled frames and the audio transcript; confirm the reasoning from visual and auditory cues.
            5. If the evidence remains unclear, do not guess. Return the most reliable answer supported by the provided data.

            ## Output Format
            Return JSON with the following structure:

            ```json
            {{
              "enumeration": [
                "Description of instance 1",
                "Description of instance 2"
              ],
              "answer": "2"
            }}
            ```

            - `enumeration` should record the enumerated items as an array of strings (order does not matter).
            - `answer` must contain only the final numeric count as a string (no units or additional text).
            """
        ).strip(),
        "placeholders": ["query_section", "frames_section", "transcription_section"],
    },
    "generate_e_to_c_openQA_with_clips": {
        "description": (
            "Adds clip evidence collection to the sequential Counting (enumerate → count) prompt."
        ),
        "template": dedent(
            """
            You are an expert annotator who analyzes long-form videos (30+ minutes) for Counting tasks.
            Follow an "enumerate first, then count" workflow and capture the evidence clips that support your answer.

            {query_section}

            {frames_section}

            {transcription_section}

            ## Counting Answer Rules
            - Basic format: numbers only (e.g., "2" or "2, 5" for multiple counts)
            - Keep answers concise; do not convert them into sentences
            - If multiple numbers are needed, separate them with commas
            - 5-second rule: repeated appearances of the same subject within 5 seconds count as one occurrence
            - Always corroborate with both visual frames and transcript cues

            ## Step-by-step Instructions
            1. Enumerate every distinct instance relevant to the query.
            2. Describe each instance with enough detail (appearance, action, timestamp hints) to distinguish it.
            3. After enumeration, deduplicate if needed and compute the exact total.
            4. Link each instance to precise evidence clips.
            5. Do not guess—return only answers supported by the provided data.

            ## Output Format
            Return JSON with the following structure:

            ```json
            {{
              "enumeration": [
                "Instance description 1",
                "Instance description 2"
              ],
              "answer": "2",
              "clips": [
                ["00:05:10", "00:05:35"],
                ["00:12:42", "00:13:05"]
              ]
            }}
            ```

            - `enumeration`: array of strings documenting each counted instance.
            - `answer`: final numeric count (string, no units or extra words).
            - `clips`: list of `[start_timestamp, end_timestamp]` pairs covering all evidence intervals.
            """
        ).strip(),
        "placeholders": ["query_section", "frames_section", "transcription_section"],
    },
    "generate_sampling_MCQ": {
        "description": (
            "Sampling frames → MCQ answer generation prompt. "
            "Template not yet finalized; fill in once specification is available."
        ),
        "template": "",
        "placeholders": [],
    },
    "generate_MCQ_question": {
        "description": (
            "Creates MCQ prompts from free-form questions and ground-truth answers. "
            "Callers should substitute the items list built from dataset rows."
        ),
        "system": dedent(
            """
            You are an expert assessment designer creating multiple-choice questions (MCQs) for video understanding tasks. You must follow the instructions strictly.
            """
        ).strip(),
        "user_template": dedent(
            """
            You will receive several question/answer pairs. For each pair, create one multiple-choice question prompt in English.

            Requirements for every item:
            1. Use the provided question verbatim.
            2. Include exactly four answer choices labeled A, B, C, D.
            3. Exactly one option must contain the exact answer string verbatim.
            4. The remaining three options must be plausible but incorrect.
            5. Shuffle the positions so the correct option is not always the same letter.
            6. The prompt must follow exactly this template (including blank lines and bullet formatting):

            Question: <question text>

            Options:
            - A. <option text>
            - B. <option text>
            - C. <option text>
            - D. <option text>

            Return your result as valid JSON with the following keys:
            - A JSON array where each element is an object containing:
              * "question_id": the provided question_id value
              * "prompt": the completed prompt string following the template
              * "correct_option": the letter (A/B/C/D) that contains the correct answer
            Do not include any additional keys or commentary.

            Items to process:
            {items_section}
            """
        ).strip(),
        "placeholders": ["items_section"],
    },
}
