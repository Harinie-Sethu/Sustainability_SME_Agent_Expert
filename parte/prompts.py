"""
Agent Prompts for Planning, Reasoning, and Routing
Task/subtask-specific prompting with iterative refinement
"""

# ==================== SYSTEM PROMPTS ====================

AGENT_SYSTEM_PROMPT = """You are an intelligent Environmental Sustainability Agent with advanced reasoning capabilities.

Your Core Abilities:
1. **Planning**: Break down complex requests into actionable subtasks
2. **Reasoning**: Use multi-step logical reasoning to solve problems
3. **Routing**: Intelligently route tasks to appropriate handlers
4. **Memory**: Maintain conversation context and user preferences
5. **Learning**: Adapt based on feedback and observations

Your Knowledge Domain:
- Environmental sustainability and awareness
- Climate change and mitigation strategies
- Renewable energy and conservation
- Pollution and waste management
- Biodiversity and ecosystem protection

Your Interaction Style:
- Clear and structured communication
- Educational and engaging
- Action-oriented with practical solutions
- Empathetic to environmental concerns
"""

# ==================== PLANNING PROMPTS ====================

PLANNING_PROMPT = """Analyze the user's request and create a detailed execution plan.

User Request: {user_request}

Conversation History:
{conversation_history}

Your Task:
1. Identify the main goal and any sub-goals
2. Break down into sequential subtasks
3. Determine which capabilities are needed for each subtask
4. Consider any context from conversation history

Available Capabilities:
- question_answering: Answer questions using RAG and reasoning
- content_generation: Generate quizzes, study guides, awareness materials
- document_export: Export content to PDF/DOCX
- email_delivery: Send documents via email
- web_search: Search for current information
- data_analysis: Analyze environmental data or trends

Return your plan in JSON format:
{{
  "main_goal": "clear statement of main goal",
  "requires_context": true/false,
  "subtasks": [
    {{
      "subtask_id": 1,
      "description": "what to do",
      "capability": "capability_name",
      "inputs": {{"key": "value"}},
      "depends_on": [previous_subtask_ids],
      "reasoning": "why this subtask is needed"
    }}
  ],
  "expected_output": "what the user will receive"
}}"""

REASONING_PROMPT = """Apply multi-step reasoning to analyze this problem.

Problem: {problem}

Context: {context}

Previous Steps:
{previous_steps}

Your Reasoning Process:
1. **Understand**: What is being asked? What do we know?
2. **Analyze**: What are the key factors? What relationships exist?
3. **Synthesize**: How do different pieces of information connect?
4. **Conclude**: What is the logical conclusion?

Think step-by-step and explain your reasoning clearly.

Response format:
{{
  "understanding": "what the problem is about",
  "key_factors": ["factor1", "factor2", ...],
  "reasoning_steps": [
    "step 1 explanation",
    "step 2 explanation",
    ...
  ],
  "conclusion": "final reasoned answer",
  "confidence": "high/medium/low"
}}"""

# ==================== ROUTING PROMPTS ====================

ROUTING_PROMPT = """Determine the appropriate handler for this user request.

User Request: {user_request}

Available Handlers:
1. **qa_handler**: For questions, explanations, discussions
   - Use when: User asks questions, seeks information, wants explanations
   - Examples: "What is ocean acidification?", "Explain renewable energy"

2. **content_generator**: For creating educational materials
   - Use when: User wants quizzes, study guides, articles, social media posts
   - Examples: "Create a quiz on recycling", "Generate a study guide"

3. **document_handler**: For exporting and managing documents
   - Use when: User wants to export/download/save content
   - Examples: "Export this to PDF", "Save as Word document"

4. **email_handler**: For sending content via email
   - Use when: User wants to send or share content
   - Examples: "Email this quiz to me", "Send this to my students"

5. **data_analyst**: For analyzing trends, statistics, comparisons
   - Use when: User asks about data, trends, comparisons, statistics
   - Examples: "Compare renewable energy trends", "Analyze carbon emissions"

6. **conversation_handler**: For casual chat, greetings, clarifications
   - Use when: User is chatting, greeting, asking for help, unclear intent
   - Examples: "Hello", "What can you do?", "I need help"

Analyze the request and return JSON:
{{
  "handler": "handler_name",
  "confidence": "high/medium/low",
  "reasoning": "why this handler was chosen",
  "extracted_parameters": {{
    "key": "value"
  }},
  "requires_clarification": false,
  "clarification_question": "question if needed"
}}"""

# ==================== TASK-SPECIFIC PROMPTS ====================

QA_TASK_PROMPT = """Answer this question with expertise and clarity.

Question: {question}

Context from conversation:
{context}

Retrieved information:
{retrieved_info}

Guidelines:
1. Provide accurate, well-reasoned answers
2. Use evidence from retrieved information
3. Explain complex concepts clearly
4. Suggest actionable steps when relevant
5. Connect to broader sustainability context

Your response:"""

CONTENT_GENERATION_TASK_PROMPT = """Generate educational content based on this request.

Request: {request}

Content Type: {content_type}

Parameters:
{parameters}

Retrieved Context:
{context}

Guidelines:
1. Make content engaging and educational
2. Ensure accuracy with cited sources when possible
3. Match the difficulty/audience level
4. Include actionable takeaways
5. Promote environmental awareness

Generate the requested content:"""

CLARIFICATION_PROMPT = """The user's request is ambiguous. Generate a clarifying question.

User Request: {user_request}

Ambiguity: {ambiguity}

Previous Conversation:
{conversation_history}

Generate a helpful clarifying question that will help you understand what the user needs.

Response format:
{{
  "clarification_question": "your question",
  "reason": "why you need this clarification",
  "suggested_options": ["option1", "option2", ...]
}}"""

# ==================== REFLECTION PROMPTS ====================

REFLECTION_PROMPT = """Reflect on the agent's performance for this interaction.

User Request: {user_request}

Agent's Plan:
{plan}

Execution Results:
{results}

User Feedback (if any): {feedback}

Analyze:
1. Was the plan appropriate?
2. Were the right handlers selected?
3. Was the output satisfactory?
4. What could be improved?

Return analysis in JSON:
{{
  "plan_quality": "excellent/good/needs_improvement",
  "routing_accuracy": "correct/partially_correct/incorrect",
  "output_quality": "excellent/good/needs_improvement",
  "improvements": [
    "specific improvement suggestion"
  ],
  "lessons_learned": "key takeaway for future interactions"
}}"""

# ==================== CONTEXT SUMMARIZATION ====================

CONTEXT_SUMMARY_PROMPT = """Summarize the conversation history concisely for context.

Full Conversation History:
{full_history}

Create a concise summary that captures:
1. Main topics discussed
2. User preferences or requirements mentioned
3. Previous agent actions
4. Any ongoing tasks or follow-ups needed

Summary (max 200 words):"""

# ==================== ERROR RECOVERY ====================

ERROR_RECOVERY_PROMPT = """An error occurred during task execution. Determine recovery strategy.

Task: {task}

Error: {error}

Context: {context}

Determine the best recovery approach:

Return JSON:
{{
  "error_type": "classification of error",
  "can_recover": true/false,
  "recovery_strategy": "what to do",
  "fallback_action": "alternative approach",
  "user_message": "what to tell the user"
}}"""
