# MTG Deck Advisor

An AI-powered Magic: The Gathering deck analysis tool that provides detailed advice for improving your decks.

## Features

- Analyze decklists and get AI-powered suggestions
- Support for all major Magic formats
- Real-time progress updates during analysis
- Downloadable markdown reports
- Card suggestions based on deck strategy

## Prerequisites

- Python 3.8 or higher
- MongoDB database
- OpenAI (or compatible) API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ari-Zerner/mtg-ai.git
cd mtg-ai
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the following variables:
```
API_KEY # API key for your LLM provider
API_BASE_URL # API base URL for your LLM provider
CHEAP_MODEL # Model for basic tasks
GOOD_MODEL # Model for complex analysis
MONGO_URI # MongoDB connection string

# Optional configuration
LOG_LEVEL # Logging level (default: INFO)
MAX_CARDS_PER_QUERY # Maximum number of cards to fetch per Scryfall query (default: 525)
MAX_CARDS_TO_CONSIDER # Maximum number of most-relevant potential additions to consider (default: 150)
MIN_RELEVANCE_SCORE # Minimum relevance score for suggestions (default: 50)
```

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter your decklist in the main text area
2. Select the format from the dropdown (optional)
3. Add any additional context or requirements (optional)
4. Click "Get Advice" to start the analysis
5. Wait for the analysis to complete
6. View and optionally download the generated report

## Development

The key components of the application are:

- `app.py`: Flask web application and job management
- `mtgai.py`: Core deck analysis and AI interaction logic
- `templates/`: HTML templates for the web interface