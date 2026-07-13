



# Writings with Code

A collection of essays on AI and software, each written as a Jupyter notebook that pairs the argument with runnable code examples.

## Publications

### Towards Reliability of AI Agents in Real-World Business Domains

An overview of research papers on the topic of **AI agent reliability**.
The essay covers τ-bench (Sierra, 2024), which made reliability measurable; and the Princeton "Science of AI Agent Reliability" work, which decomposed it into four dimensions (consistency, robustness, predictability, safety);
Additionally touches practical aspects of agent development lifecycle (ADLC) -- how to integrate agent evals into the development lifecycle.

Notebook: [`towards_reliable_ai_agents.ipynb`](towards_reliable_ai_agents.ipynb)

## Running the notebook

```bash
uv run --with jupyterlab jupyter lab towards_reliable_ai_agents.ipynb
```

## Converting to static HTML

Render the notebook (with its outputs) into a self-contained HTML file:

```bash
uv run --with nbconvert jupyter nbconvert --to html towards_reliable_ai_agents.ipynb
```

This writes `towards_reliable_ai_agents.html`. To embed images and styles inline for a single portable file, add `--embed-images`. To re-execute every cell before exporting so the outputs are fresh, add `--execute`.

