import asyncio
from agent_first_browse.agent.graph import run_brain

objective = """
MISSION: READ-ONLY HUMAN BROWSING TEST

TARGET PLATFORM: https://www.reddit.com/r/rust
OBJECTIVE: Navigate to the subreddit, scroll around to simulate a human reading, and then finish.

CRITICAL RULES FOR THIS RUN:
1. STRICTLY READ-ONLY: DO NOT click "Create Post", DO NOT type anything, DO NOT submit any forms, and DO NOT log in if prompted.
2. HUMAN BEHAVIOR: Just navigate to the URL, scroll down the page 2 or 3 times to simulate reading the feed, and then output action_type="done".
"""

if __name__ == "__main__":
    print("Starting Read-Only Browsing Test on Reddit...")
    asyncio.run(run_brain(objective))
