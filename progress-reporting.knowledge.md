# Progress Reporting System

## Overview
The deck analysis process reports progress through a callback system. Progress messages flow from mtgai.py functions through app.py to the frontend.

## Key Components
- Backend functions accept optional progress_callback parameter
- Jobs dictionary in app.py stores progress messages
- Frontend polls /status endpoint for updates

## Progress Message Flow
1. mtgai.py functions call progress_callback
2. app.py's progress_update adds message to job record
3. /status endpoint returns messages to frontend
4. progress.html displays messages and polls for updates

## Customizing Progress Messages
To modify progress messages:

1. In mtgai.py:
   - get_deck_advice: Overall deck analysis progress
   - get_potential_additions: Card search progress
   - get_card_descriptions_dict: Card fetching progress

2. In app.py:
   - run_job: Job start/completion messages
   - Error handling messages

Example progress points:
```python
# In get_deck_advice:
progress_callback("Starting deck analysis...")
progress_callback(f"Extracted {len(cards)} card names")
progress_callback("Fetching card descriptions...")

# In get_potential_additions:
progress_callback("Analyzing deck strategy...")
progress_callback("Searching for potential additions...")
progress_callback("Evaluating card suggestions...")
```
