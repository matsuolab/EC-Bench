PROMPT = """
You are a dataset annotator who specializes in counting tasks for long-form videos (30 minutes or longer).
You will receive one video. Inspect the visuals and audio carefully and craft twelve high-quality queries **in English** across the six counting categories described below. Provide the corresponding answers in English as well.

## Six Counting Categories

### A. Parallel Event Counting
**Definition:** Count discrete events that occur independently along the timeline. Semantic event recognition is the focus.
- **Counting example:** “How many scenes show self-driving trucks cruising on the highway?”
- **Enumeration example:** “List every type of flying vehicle that appears.”

### B. Causal Event Counting
**Definition:** Count ordered or causally linked events. Requires procedural understanding and temporal reasoning.
- **Counting example:** “How many shots occur in the rally that starts at 4:15?”
- **Enumeration example:** “List the steps in the facial-recognition payment workflow.”

### C. Speech & Audio Counting
**Definition:** Count audio-based events or spoken content (multimodal counting).
- **Counting example:** “How many times does a siren sound?”
- **Enumeration example:** “List the industries mentioned by the lecturer.”

### D. Appearance Counting
**Definition:** Count how frequently specific objects, locations, or shapes appear. Only information visible in the video qualifies.
- **Counting example:** “How many times do players wearing blue uniforms appear on screen?”
- **Enumeration example:** “List the creatures shaped like constellations that appear.”

### E. Spatial Counting
**Definition:** Count how many people or objects exist within a single frame or scene. Must be visually verifiable.
- **Counting example:** “How many bicycles are on the road simultaneously?”
- **Enumeration example:** “List the future medical technologies that are shown.”

### F. Spatio-Temporal Conditional Counting
**Definition:** Count events that satisfy both spatial and temporal conditions. Evaluates context-heavy, long-form reasoning.
- **Counting example:** “How many shots are taken inside the penalty area during the second half?”
- **Enumeration example:** “List the changes compared to 50 years ago that are presented in the video.”

## Answering Rules

### 1. Counting Queries
**Definition:** Questions answered with numerical values (counts, people, ratios, etc.).
**Formatting rules:**
- Output only the numeric value (optionally include a unit).
- Keep it concise; provide numbers without prose.
- When multiple numbers are required, separate them with commas (e.g., “2, 5, 7”).
- Single-count rule: When counting occurrences of a specific object, treat repeated appearances within five seconds as one occurrence; gaps longer than five seconds count as separate occurrences.
**Examples:** “4回”, “12人”, “3本”

### 2. Enumeration Queries
**Definition:** Questions that require listing multiple entities or steps.
**Formatting rules:**
- Provide comma-separated lists.
- Use “→” to connect steps when order matters.
- Use placeholder expressions (e.g., “Person A,” “Male,” “Female”) if proper nouns are unknown.
- If something belongs to someone, omit the owner’s name unless critical (“A’s car” → “車”).
- No extra commentary—list only the required items or steps.
**Examples:**
- Simple list: “バット、グローブ、ボール”
- People list: “選手A、選手B、選手C”
- Procedural list: “鍛造→研磨→装飾→完成”

### 3. Supporting Clip Generation
**Definition:** Identify the video clips that justify the answer.
**Formatting rules:**
- Use the format [start_time, end_time], e.g., [0:04:45, 0:04:51].
- Seconds-level granularity is fine; frame-accurate timestamps are unnecessary.
- List all clips if multiple segments support the answer (e.g., [0:04:45, 0:04:51], [0:31:45, 0:31:55]).
- Merge clips into a single entry if the interval between them is under five seconds.
- When boundaries are ambiguous, include a small buffer before and after the relevant moment.
**Selection criteria:**
- Extract the minimal set of clips required to answer the query.
- For comparison questions, include clips for every item being compared.
- For ratio questions, include clips covering all values used in the calculation.
- When repeated appearances occur, capture only the minimal footage needed for verification.
**Examples:**
- Single clip: [0:15:30, 0:15:45]
- Multiple clips: [0:04:45, 0:04:51], [0:31:45, 0:31:55], [0:58:20, 0:58:30]

## Global Constraints (All Categories)
- **Whole-video understanding:** Queries must require knowledge of the entire video, not just the intro or a short moment.
- **Video/audio dependence:** Avoid questions that can be answered purely with common sense.
- **Multiple evidence points:** Prefer queries that require citing two or more clips.
- **Objectivity:** Questions must be answerable with factual evidence, not personal opinions.
- **Visual dependence:** Except for category C, questions must be solvable using visual information alone.
- **Visual verification:** Avoid relying on prior knowledge of proper nouns; use what is visible.
- **Concise single-line queries:** No prefixes or numbering; keep phrasing clear and short.

## Sports-Specific Templates

### Baseball
**Category A:** Counting—“How many hits were recorded?”; “How many home runs occurred?”  
Enumeration—“List all hit types that produced runs.”

**Category B:** Counting—“How many pitches occurred before the home run?”  
Enumeration—“List the sequence of plate appearances leading to the grand slam.”

**Category C:** Counting—“How many times did the umpire make a call?”  
Enumeration—“List every team name mentioned by the commentator.”

**Category D:** Counting—“How many times do players in blue uniforms appear?”  
Enumeration—“List the jersey numbers of every pitcher who appears.”

**Category E:** Counting—“What is the maximum number of baserunners on simultaneously?”  
Enumeration—“List the bench players’ jersey numbers in the bottom of the ninth.”

**Category F:** Counting—“How many hits occur from the seventh inning onward when runners are on base?”  
Enumeration—“List all extra-inning moments with runners in scoring position.”

### Soccer
**Category A:** Counting—“How many shots did Team A take?”; “How many yellow cards were issued?”  
Enumeration—“List the types of plays that led to goals.”

**Category B:** Counting—“How many passes were made in front of the goal?”  
Enumeration—“List the attacking patterns that produced the comeback goal.”

**Category C:** Counting—“How many times did the referee whistle?”  
Enumeration—“List every tactical term mentioned in commentary.”

**Category D:** Counting—“How many times do players in white uniforms appear?”  
Enumeration—“List the jersey numbers of substitutes who entered the pitch.”

**Category E:** Counting—“What is the maximum number of players inside the penalty area?”  
Enumeration—“List the jersey numbers present in the box during corner kicks.”

**Category F:** Counting—“How many shots occurred inside the penalty area during the second half?”  
Enumeration—“List every penalty-area play during stoppage time.”

### Basketball
**Category A:** Counting—“How many three-pointers did Team A make?”; “How many fouls were called?”  
Enumeration—“List every scoring pattern that appears.”

**Category B:** Counting—“How many successful fast breaks followed steals?”  
Enumeration—“List the sequence of plays that produced the comeback.”

**Category C:** Counting—“How many times did the referee blow the whistle?”  
Enumeration—“List the instructions shouted by the coach.”

**Category D:** Counting—“How many times do players in red uniforms appear?”  
Enumeration—“List the jersey numbers of all substitutes.”

**Category E:** Counting—“What is the maximum number of players simultaneously in the paint?”  
Enumeration—“List the jersey numbers on the court at the start of Q4.”

**Category F:** Counting—“How many successful paint-area shots occur in Q4?”  
Enumeration—“List every paint-area play within the final two minutes.”

**Important:** For sports videos, incorporate sport-specific statistical or tactical elements inspired by the templates above. Tailor each category to the sport featured in the footage.

## Query Requirements
- Produce **12 total queries**.
- Cover **all six categories (A–F)**.
- Provide **one Counting and one Enumeration query per category**.
- Ensure each query depends on multiple sections of the video and cannot be answered without watching the footage.
- **Write both the query text and the answer text in English.**

## Output Format
Return only the JSON below:

```json
{
  "queries": [
    {
      "category": "A",
      "tag": "Counting",
      "query": "...",
      "answer": "...",
      "clips": ["[...]", "[...]"]
    },
    {
      "category": "A",
      "tag": "Enumeration",
      "query": "...",
      "answer": "...",
      "clips": ["[...]"]
    },
    [...],
    {
      "category": "F",
      "tag": "Counting",
      "query": "...",
      "answer": "...",
      "clips": ["[...]", "[...]"]
    },
    {
      "category": "F",
      "tag": "Enumeration",
      "query": "...",
      "answer": "...",
      "clips": ["[...]"]
    }
  ]
}
```

**Clip formatting examples**
- Single clip: `["[0:15:30, 0:15:45]"]`
- Multiple clips: `["[0:04:45, 0:04:51]", "[0:31:45, 0:31:55]"]`

Craft diverse, practical counting queries that leverage the nuances of the video’s content (sports, documentary, vlog, etc.) while following every rule above.
"""
