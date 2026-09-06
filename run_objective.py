import asyncio
from agent_first_browse.agent.graph import run_brain

if __name__ == '__main__':
    with open('objective.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()
    asyncio.run(run_brain(prompt))
