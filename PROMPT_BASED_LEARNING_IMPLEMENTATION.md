# Prompt-Based Learning Implementation in Environmental SME Project

This document shows **where**, **what**, and **how** prompt-based learning is implemented in this project.

---

## 📍 **WHERE: Location of Implementation**

### 1. **Core Prompt Engineering System** (`partf/`)
- **`partf/prompt_library.py`**: Central library of prompt templates organized by task
- **`partf/few_shot_examples.py`**: Few-shot example collections for different tasks
- **`partf/prompt_engineer.py`**: Main prompt engineering system with testing and iteration
- **`partf/experiment_runner.py`**: Orchestrates systematic experiments
- **`partf/ablation_studies.py`**: Studies impact of removing prompt components
- **`partf/failure_analyzer.py`**: Analyzes failures to improve prompts

### 2. **Agent-Specific Prompts** (`parte/`)
- **`parte/prompts.py`**: Flow-specific prompts for planning, reasoning, routing, and task execution

### 3. **Task Handlers** (`partd/`)
- **`partd/task_handlers.py`**: Multi-step chain prompting for quiz generation and content creation
- **`partd/enhanced_rag.py`**: Adaptive prompting with feedback integration

### 4. **Storage Directories**
- **`partf/experiments/prompts/`**: Saved prompt templates (JSON format)
- **`partf/experiments/few_shots/`**: Saved few-shot examples (JSON format)
- **`partf/experiments/results/`**: Experiment results and evaluations (JSON format)

---

## 🎯 **WHAT: Types of Prompt-Based Learning**

### 1. **Effective Prompt Design**

#### **A. Prompt Library Structure** (`partf/prompt_library.py`)
Organized by task with multiple variants:

```python
# Task categories:
- qa (Question Answering)
- quiz_generation
- study_guide
- awareness_content
- reasoning
- summarization

# Variants per task (example for Q&A):
- basic: Simple zero-shot
- contextual: With context injection
- chain_of_thought: Step-by-step reasoning
- structured: Formatted output
- with_confidence: Includes confidence levels
```

**Example from code:**
```python
"chain_of_thought": {
    "version": "1.0",
    "template": """You are an expert in Environmental Sustainability.

Question: {question}

Let's think step by step:
1. First, understand what is being asked
2. Identify the key concepts involved
3. Connect relevant information
4. Provide a comprehensive answer

Answer:""",
    "description": "Q&A with chain-of-thought reasoning",
    "parameters": ["question"]
}
```

#### **B. Flow-Specific Prompting** (`parte/prompts.py`)
Different prompts for different agent flows:

- **Planning Prompt**: Breaks down complex requests into subtasks
- **Reasoning Prompt**: Multi-step logical reasoning
- **Routing Prompt**: Determines appropriate handler
- **Task-Specific Prompts**: QA, content generation, clarification, reflection

**Example:**
```python
PLANNING_PROMPT = """Analyze the user's request and create a detailed execution plan.

User Request: {user_request}

Conversation History:
{conversation_history}

Your Task:
1. Identify the main goal and any sub-goals
2. Break down into sequential subtasks
3. Determine which capabilities are needed for each subtask
...
"""
```

### 2. **Few-Shot Prompting**

#### **A. Few-Shot Library** (`partf/few_shot_examples.py`)
Structured examples organized by task:

```python
# Task categories with examples:
- qa: 3 Q&A examples (renewable energy, composting, ocean acidification)
- quiz: 3 quiz examples (solar energy, recycling, climate change)
- summarization: 2 summarization examples
- reasoning: 2 reasoning examples
```

**Example structure:**
```python
{
    "input": "What is renewable energy?",
    "output": "Renewable energy is energy derived from natural sources..."
}
```

#### **B. Few-Shot Formatting** (`partf/few_shot_examples.py:180-206`)
Automatically formats prompts with examples:

```python
def format_few_shot_prompt(self, task: str, num_examples: int, new_input: Any) -> str:
    """
    Format a few-shot prompt with examples.
    
    Returns formatted prompt like:
    Here are some examples:
    
    Example 1:
    Input: ...
    Output: ...
    
    Now process this input:
    Input: {new_input}
    Output:
    """
```

### 3. **Iterative Prompt Correction/Adaptation**

#### **A. Iterative Refinement** (`partf/prompt_engineer.py:306-370`)
Systematically refines prompts based on results:

```python
def iterative_refinement(self, task: str, initial_variant: str,
                       test_cases: List[Dict], max_iterations: int = 3):
    """
    Iteratively refine prompt based on results.
    
    Process:
    1. Test current variant on test cases
    2. Analyze success rate and match scores
    3. If performance < threshold:
       - Analyze failures
       - Generate refinement suggestions
       - Update prompt
    4. Repeat until performance is acceptable
    """
```

**Key features:**
- Tracks refinement history
- Analyzes failures to suggest improvements
- Stops when success rate ≥ 90% and match score ≥ 80%

#### **B. Failure Analysis** (`partf/failure_analyzer.py`)
Analyzes failures to improve prompts:

```python
def _analyze_failures(self, failures: List[Dict]) -> List[str]:
    """
    Analyze failure patterns and suggest improvements.
    
    Checks for:
    - JSON parsing errors → Add explicit JSON format instructions
    - Short outputs → Add instruction to be more detailed
    - Timeouts → Reduce prompt complexity
    """
```

#### **C. Self-Learning from Feedback** (`partd/enhanced_rag.py:360-391`)
Integrates user feedback into prompts:

```python
def _get_feedback_context(self, question: str) -> List[Dict]:
    """Load relevant feedback to improve answers."""
    
def _generate_adaptive_answer(self, question: str, context: str,
                              feedback_context: List[Dict], ...):
    """
    Generate adaptive answer with feedback integration.
    
    Includes learned improvements in system prompt:
    - "Learned improvements from feedback: ..."
    """
```

### 4. **Multi-Step Chain Prompting** (`partd/task_handlers.py:93-168`)
Sequential prompts for complex tasks:

```python
# Quiz generation uses 3-step chain:
chain_steps = [
    {
        "name": "analyze_topic",
        "prompt": "Analyze the topic for quiz generation...",
        "output_key": "analysis"
    },
    {
        "name": "generate_questions",
        "prompt": "Generate questions using {analysis}...",
        "output_key": "questions_raw"
    },
    {
        "name": "enhance_explanations",
        "prompt": "Enhance explanations using {questions_raw}...",
        "output_key": "final_quiz"
    }
]
```

---

## 🔬 **HOW: Experimentation and Learning**

### 1. **Systematic Experimentation**

#### **A. Prompt Variant Comparison** (`partf/prompt_engineer.py:150-209`)
Compares multiple prompt variants on same test cases:

```python
def compare_variants(self, task: str, variants: List[str], 
                    test_cases: List[Dict]) -> Dict[str, Any]:
    """
    Compare multiple prompt variants.
    
    Returns:
    - Results for each variant
    - Summary statistics (success rate, execution time, match score)
    - Best variant identification
    """
```

#### **B. Few-Shot Experiments** (`partf/prompt_engineer.py:211-257`)
Tests different numbers of examples:

```python
def test_few_shot(self, task: str, num_shots: int, test_input: Any, ...):
    """
    Test few-shot prompting with different numbers of examples.
    
    Tests: 0-shot, 1-shot, 3-shot, 5-shot, etc.
    """
```

#### **C. Ablation Studies** (`partf/ablation_studies.py`)
Systematically removes components to measure impact:

```python
def study_instruction_removal(self, base_prompt: str, ...):
    """
    Study impact of removing instructions from prompt.
    
    Compares:
    - Baseline (full prompt)
    - Ablated (without instructions)
    - Measures quality degradation
    """
```

### 2. **Structured Organization (JSON Format)**

#### **A. Prompt Storage** (`partf/prompt_library.py:444-449`)
Saves prompts to JSON files:

```python
def save_prompts(self):
    """Save prompt library to disk."""
    for task, variants in self.prompts.items():
        filepath = self.storage_dir / f"{task}_prompts.json"
        with open(filepath, 'w') as f:
            json.dump(variants, f, indent=2)
```

**Example JSON structure:**
```json
{
  "qa": {
    "basic": {
      "version": "1.0",
      "template": "...",
      "description": "...",
      "parameters": [...]
    },
    "chain_of_thought": {...}
  }
}
```

#### **B. Few-Shot Storage** (`partf/few_shot_examples.py:208-213`)
Saves examples to JSON:

```python
def save_examples(self):
    """Save examples to disk."""
    for task, examples in self.examples.items():
        filepath = self.storage_dir / f"{task}_examples.json"
        with open(filepath, 'w') as f:
            json.dump(examples, f, indent=2)
```

**Example JSON structure:**
```json
[
  {
    "input": "What is renewable energy?",
    "output": "Renewable energy is..."
  },
  {
    "input": "How does composting help?",
    "output": "Composting benefits..."
  }
]
```

#### **C. Experiment Results** (`partf/prompt_engineer.py:444-473`)
Exports all experiments to JSON:

```python
def export_experiments(self, filepath: Optional[str] = None) -> str:
    """
    Export all experiment results.
    
    Structure:
    {
      "total_experiments": N,
      "experiments": [...],
      "summary": {
        "success_rate": ...,
        "avg_execution_time": ...
      }
    }
    """
```

**Example result file:** `partf/experiments/results/test_experiments_20251117_132354.json`

### 3. **Performance Tracking**

#### **A. Performance Logging** (`partf/prompt_library.py:451-459`)
Tracks prompt performance:

```python
def log_performance(self, task: str, variant: str, metrics: Dict[str, Any]):
    """Log prompt performance."""
    log_entry = {
        "timestamp": ...,
        "task": task,
        "variant": variant,
        "metrics": metrics
    }
    self.performance_log.append(log_entry)
```

#### **B. Experiment Tracking** (`partf/prompt_engineer.py:47-48`)
Tracks all experiments and iterations:

```python
self.experiments: List[Dict] = []  # All experiments
self.iterations: Dict[str, List[Dict]] = {}  # Iteration history
```

---

## 📊 **Key Findings from Experiments**

### 1. **Prompt Variants Tested**
- **Q&A**: basic, contextual, chain_of_thought, structured, with_confidence
- **Quiz Generation**: basic, difficulty_aware, bloom_taxonomy, scenario_based
- **Study Guides**: comprehensive, modular, visual_focused
- **Awareness Content**: article, social_media, storytelling

### 2. **Few-Shot Configurations**
- Tested: 0-shot, 1-shot, 3-shot, 5-shot
- Organized by task with structured input/output pairs

### 3. **Iterative Refinement Process**
1. Test current prompt variant
2. Calculate success rate and match scores
3. If performance < threshold:
   - Analyze failures
   - Generate refinement suggestions
   - Update prompt
4. Repeat until performance acceptable

### 4. **Structured Evaluation**
- All experiments saved as JSON pairs (input/output)
- Performance metrics tracked systematically
- Failure analysis generates improvement suggestions

---

## 🔄 **Integration Points**

### 1. **RAG System** (`partd/enhanced_rag.py`)
- Uses adaptive prompting with feedback
- Integrates learned improvements into system prompts

### 2. **Multi-Agent System** (`parte/`)
- Uses flow-specific prompts for planning, reasoning, routing
- Task-specific prompts for different handlers

### 3. **Content Generation** (`partd/task_handlers.py`)
- Uses multi-step chain prompting
- Each step builds on previous output

---

## 📝 **Summary**

This project implements comprehensive prompt-based learning through:

1. **Organized Prompt Library**: Centralized, versioned prompts by task and variant
2. **Few-Shot Examples**: Structured input/output pairs in JSON format
3. **Iterative Refinement**: Systematic testing and improvement based on results
4. **Experimentation**: Variant comparison, ablation studies, few-shot testing
5. **Structured Storage**: All prompts, examples, and results saved as JSON
6. **Performance Tracking**: Metrics logged for evaluation and improvement
7. **Flow-Specific Prompting**: Different prompts for different agent workflows
8. **Self-Learning**: Feedback integration for continuous improvement

All components work together to systematically identify which prompt structures, example selections, and chaining approaches produce reliable results for the Environmental Sustainability domain.

