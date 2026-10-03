# Agentic AI Fundamentals

## From LLM Calls to Reliable Tool-Using Agents

This tutorial is a notebook-based introduction to building agentic AI systems from first principles. It begins with the difference between an ordinary language-model call and an agent, then gradually introduces structured outputs, safe tools, native function calling, agent loops, memory, LangGraph, agentic RAG, approval workflows, evaluation, multi-agent coordination, MCP, and production architecture.

The course is designed for readers who can write basic Python but have not built an agent before. Each notebook introduces one architectural idea, explains why it exists, and develops it through code before the next abstraction is added.

The final result is not simply a chatbot with several tools. It is a small but complete agent system with typed contracts, retrieval, policy enforcement, human approval, persistence, MCP integration, idempotent side effects, traces, and end-to-end evaluation.

---

## Who this tutorial is for

This series is suitable for:

- Python developers beginning to explore agentic AI;
- data scientists moving from model experiments toward AI systems;
- AI engineers who want to understand agent runtimes beneath high-level frameworks;
- students looking for a structured path through tools, RAG, LangGraph, and MCP; and
- readers who have seen agent demos but want to learn how reliable implementations are designed.

You do not need previous LangGraph or MCP experience. Familiarity with Python functions, dictionaries, classes, and basic API usage will be helpful.

---

## What you will build

Across the notebooks, you will create:

- a decision model for choosing between an LLM call, chain, workflow, and agent;
- validated model outputs using Pydantic;
- a typed tool registry with controlled execution;
- a real Gemini function-calling integration;
- a reusable and bounded agent loop;
- explicit short-term memory and context management;
- stateful agents with LangGraph;
- retrieval exposed as an agent tool;
- layered guardrails and human approval;
- an agent evaluation and regression harness;
- a coordinated multi-agent team;
- a real MCP server and client; and
- a production-minded support-agent capstone.

---

## Course roadmap

The notebooks are intended to be completed in order. Later notebooks reuse concepts and terminology established earlier in the course.

| # | Notebook | Main idea | API required? |
|---:|---|---|:---:|
| 00 | [Agentic AI Mental Model](./00_Agentic_AI_Mental_Model.ipynb) | Understand what makes a system agentic and when an agent is appropriate | No |
| 01 | [LLM, Chain, Workflow, or Agent?](./01_LLM_Chain_Workflow_or_Agent.ipynb) | Choose the smallest suitable architecture for a task | No |
| 02 | [Structured Outputs with Pydantic](./02_Structured_Outputs_with_Pydantic.ipynb) | Convert probabilistic model output into validated application data | No |
| 03 | [Building Safe Tools and a Tool Registry](./03_Building_Safe_Tools_and_a_Tool_Registry.ipynb) | Separate tool definitions, validation, registration, and execution | No |
| 04 | [Real Gemini API and Native Function Calling](./04_Real_Gemini_API_and_Native_Function_Calling.ipynb) | Let a real model request tools without giving it execution control | Yes |
| 05 | [Building a Reusable Agent Loop](./05_Building_a_Reusable_Agent_Loop.ipynb) | Support repeated model/tool turns with explicit limits and termination | Optional |
| 06 | [Agent State, Memory, and Context Management](./06_Agent_State_Memory_and_Context_Management.ipynb) | Manage conversation state without treating all history as permanent memory | Optional |
| 07 | [LangGraph Fundamentals and Stateful Agents](./07_LangGraph_Fundamentals_and_Stateful_Agents.ipynb) | Represent agent execution as nodes, edges, state, and checkpoints | Optional |
| 08 | [Agentic RAG: Retrieval as a Tool](./08_Agentic_RAG_Retrieval_as_a_Tool.ipynb) | Allow an agent to decide when retrieval is necessary | Optional |
| 09 | [Guardrails, Human Approval, and Failure Recovery](./09_Guardrails_Human_Approval_and_Failure_Recovery.ipynb) | Protect tool execution with policy, approval, retries, and redaction | Optional |
| 10 | [Agent Evaluation, Tracing, Cost, and Latency](./10_Agent_Evaluation_Tracing_Cost_and_Latency.ipynb) | Evaluate final outcomes, trajectories, safety, reliability, and operations | Optional |
| 11 | [Multi-Agent Delegation, Handoffs, and Coordination](./11_Multi_Agent_Delegation_Handoffs_and_Coordination.ipynb) | Coordinate specialized agents through explicit contracts and state | Optional |
| 12 | [MCP Fundamentals: Tools, Resources, and Prompts](./12_MCP_Fundamentals_Tools_Resources_and_Prompts.ipynb) | Build and use a real MCP server and client with the stable Python SDK | Optional |
| 13 | [Production Agent Architecture and Capstone](./13_Production_Agent_Architecture_and_Capstone.ipynb) | Combine retrieval, planning, approval, MCP, persistence, and evaluation | Optional |

“Optional” means that the notebook contains a complete local path that can be studied without an API key, followed by additional cells that use Gemini.

---

## Learning progression

The course is organized into four stages.

### Stage 1 — Foundations

Notebooks 00–03 establish the boundaries between models and application code. You will learn why agents require structured contracts and why a model requesting a function is different from the application executing it.

### Stage 2 — Building a stateful agent

Notebooks 04–08 introduce real function calling, reusable agent loops, memory, LangGraph, and retrieval. By the end of this stage, the agent can decide when to use external information and can carry explicit state through a graph.

### Stage 3 — Reliability and coordination

Notebooks 09–11 focus on the engineering work that demo agents often omit: approval, bounded retries, idempotency, traces, regression evaluation, cost, latency, delegation, and handoffs.

### Stage 4 — Integration and production architecture

Notebooks 12–13 introduce MCP and combine the full course into a capstone with persistent checkpoints and controlled side effects.

---

## Core architecture

The later notebooks build toward the following execution model:

                         ┌────────────────────────┐
                         │      User request      │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │    Input validation    │
                         └───────────┬────────────┘
                                     │                  ┌────────────────────────┐
                                     ├──── events ────► │ Tracing and evaluation │
                                     │                  └────────────────────────┘
                                     │                  
                                     ▼
                         ┌────────────────────────┐
                         │ Retrieval and context  │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ Model action proposal  │
                         └───────────┬────────────┘
                                     │                  ┌────────────────────┐
                                     ├──── state ─────► │ Checkpoint storage │
                                     │                  └────────────────────┘
                                     │                  
                                     ▼
                         ┌────────────────────────┐
                         │      Policy gate       │
                         └───────────┬────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     │               │               │
                     ▼               ▼               ▼
                Read only       Side effect        Denied
                     │               │               │
                     ▼               ▼               │
          ┌───────────────────┐ ┌────────────────┐   │
          │ Tool / MCP        │ │ Human approval │   │
          │ execution         │ └───────┬────────┘   │
          └─────────▲─────────┘         │            │
                    │                   │            │
                    └───────────────────┘            │
                    │                                │
                    ▼                                │
          ┌─────────────────────┐                    │
          │ Controlled response │◄───────────────────┘
          └─────────────────────┘

The model proposes actions, while the application owns validation, authorization, execution, persistence, and observability.

---

## Running the notebooks in Google Colab

1. Open a notebook from the roadmap above.
2. Use **Open in Colab** from GitHub, or upload the notebook directly to Colab.
3. Run the cells from top to bottom.
4. Complete the exercise before revealing any provided solution or guidance.
5. Use a fresh runtime when moving between notebooks if installed package versions conflict.

Every notebook is self-contained. The tutorial does not rely on external project `.py` files, so the notebooks can be studied independently in Colab.

### Configuring Gemini

The local examples run without paid model calls. For cells marked as requiring Gemini:

1. create a Gemini API key in Google AI Studio;
2. open **Secrets** in the Colab sidebar;
3. add a secret named `GEMINI_API_KEY`; and
4. enable notebook access to that secret.

The notebooks read the key from Colab Secrets and do not print it. Avoid placing API keys directly inside notebook cells or committing them to Git.

---

## Main technologies

- **Python** for the runtime and deterministic application logic
- **Pydantic** for validation and structured contracts
- **Google Gemini API** for real model and function-calling examples
- **LangGraph** for stateful orchestration, interrupts, and checkpoints
- **MCP Python SDK v2** for tools, resources, prompts, and capability discovery
- **SQLite** for the capstone's checkpoints and idempotent business records
- **Google Colab** as the primary learning environment

---

## Engineering principles used throughout

The notebooks repeatedly return to a small set of principles:

1. **Use the simplest suitable architecture.** Not every task requires an agent.
2. **Treat model output as untrusted input.** Validate it before using it.
3. **Separate proposal from execution.** A tool call suggested by a model is not authorization.
4. **Keep side effects behind policy and approval.** Risk belongs to the application boundary.
5. **Make state and termination explicit.** Agent loops must have observable stopping conditions.
6. **Preserve provenance.** Retrieved evidence and tool observations should remain traceable.
7. **Design for retries.** Side-effecting tools should be idempotent.
8. **Evaluate trajectories as well as answers.** A persuasive final response can hide an unsafe path.
9. **Compare complexity with a simpler baseline.** Multi-agent systems must justify their extra cost.
10. **Treat production readiness as a system property.** A strong model is only one component.

---

## Exercises and validation

Each notebook includes practical checks or an exercise. The supplied notebooks were validated by executing their local code paths, while API-dependent cells were kept separate so readers can run them with their own credentials.

The final capstone includes tests for:

- grounded read-only responses;
- approval interrupts;
- persisted checkpoint state;
- approved tool execution;
- rejected actions;
- source preservation;
- MCP invocation; and
- duplicate side-effect prevention.

The final exercise asks you to add a complete read-only capability across the MCP, schema, routing, policy, response, and evaluation layers. It is intentionally left unsolved because implementing a full vertical slice is the best test of whether the architecture is understood.

---

## Important limitations

These notebooks teach architecture in a controlled environment. They are not a drop-in production service. A deployed agent would additionally require authenticated users, tenant isolation, secrets management, database migrations, protected logs, remote-service authentication, distributed tracing, rate limits, timeouts, concurrency testing, incident response, and privacy review.

Model behavior and provider APIs also evolve. Keep dependency versions controlled, review provider migration notes, and rerun the evaluation suite whenever a model, prompt, tool schema, or orchestration rule changes.

---

## Completion outcome

After finishing the course, you should be able to explain and implement the path from a single LLM call to a controlled tool-using agent. More importantly, you should be able to reason about where reliability comes from: typed interfaces, narrow permissions, explicit state, durable checkpoints, human oversight, observable execution, and repeatable evaluation.

The course contains **14 notebooks, numbered 00–13**, and is complete.

