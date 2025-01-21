import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import pymongo
import certifi
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
def env_var(name):
    value = os.getenv(name)
    if not value:
        logger.error(f"Environment variable {name} not found")
        raise ValueError(f"{name} is not set in the environment variables")
    logger.debug(f"Loaded environment variable: {name}")
    return value

load_dotenv()
logger.info("Loading environment variables")
DEEPSEEK_API_KEY = env_var("DEEPSEEK_API_KEY")
MONGO_URI = env_var("MONGO_URI")

def get_card_json(card_name):
    """
    Search for a card using the Scryfall API and return the first result
    
    Args:
        card_name (str): Name of the card to search for
        
    Returns:
        dict: Card data from Scryfall API
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
        ValueError: If no cards are found
    """
    logger.info(f"Searching Scryfall for card: {card_name}")
    url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    if data["total_cards"] == 0:
        logger.error(f"No cards found matching: {card_name}")
        raise ValueError(f"No cards found matching '{card_name}'")
    
    logger.debug(f"Found {data['total_cards']} matching cards")    
    return data["data"][0]

openai = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/")

def generate_card_description(card_json):
    """
    Get a natural language description of a Magic card's functional aspects using GPT-4
    
    Args:
        card_name (str): Name of the card to describe
        
    Returns:
        str: Natural language description of the card's functionality
        
    Raises:
        requests.exceptions.RequestException: If the Scryfall API request fails
        ValueError: If no cards are found
        openai.OpenAIError: If the OpenAI API request fails
    """
    logger.info(f"Generating description for card: {card_json['name']}")
    
    # Construct prompt for GPT
    system_prompt = """You are an automatic system that parses JSON representations of Magic: The Gathering cards.
    You return plain text detailing all mechanics of the card, and no other information.
    You are familiar with concise representations of card mechanics such as mana symbols.
    Your knowledge of the rules enables you to understand which information is relevant based on the card.
    You don't interpret the card, you just put the information in plain text.
    You are as concise as possible and don't need to use full sentences or proper grammar.
    You always provide all functional information about the card, even if this risks including a small amount of irrelevant information.
    If a card has two faces, you separate the two faces with a double slash (//). A card's description never includes a double newline.
    
    Example outputs:
    
    
    Lightning Bolt
    {R}
    Instant
    Deal 3 damage to any target.
    
    
    Baleful Strix
    {U}{B}
    Artifact Creature - Bird
    1/1
    Flying, deathtouch
    When Baleful Strix enters, draw a card.
    
    
    Teferi, Hero of Dominaria
    {3}{W}{U}
    Legendary Planeswalker - Teferi
    Loyalty: 4
    +1: Draw a card. At the beginning of the next end step, untap up to two lands.
    -3: Put target nonland permanent into its owner's library third from the top.
    -8: You get an emblem with "Whenever you draw a card, exile target permanent an opponent controls."


    Invasion of Ixalan
    {1}{G}
    Battle — Siege
    Defense: 4
    When Invasion of Ixalan enters, look at the top five cards of your library. You may reveal a permanent card from among them and put it into your hand. Put the rest on the bottom of your library in a random order.
    //
    Belligerent Regisaur
    Creature — Dinosaur
    4/3
    Trample
    Whenever you cast a spell, Belligerent Regisaur gains indestructible until end of turn.
    
    
    Razorgrass Ambush
    {1}{W}
    Instant
    Razorgrass Ambush deals 3 damage to target attacking or blocking creature.
    //
    Razorgrass Field
    Land
    As Razorgrass Field enters, you may pay 3 life. If you don't, it enters tapped.
    {T}: Add {W}.
    """
    
    logger.debug("Making OpenAI API call for card description")
    # Get description from GPT
    response = openai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{card_json}"}
        ],
        temperature=0.1
    )
    logger.debug("Successfully received response from OpenAI")
    return response.choices[0].message.content

def get_card_descriptions_dict(card_names, name_to_json={}):
    """
    Get natural language descriptions of Magic cards' functional aspects,
    first checking MongoDB in bulk and falling back to Scryfall+AI for missing cards.
    
    Args:
        card_names (list): List of card names to describe
        name_to_json (dict): Optional dictionary mapping card names to their JSON representations, to avoid redundant Scryfall API requests
    Returns:
        dict: Dictionary mapping card names to their descriptions
        
    Raises:
        requests.exceptions.RequestException: If any Scryfall API requests fail
        ValueError: If no cards are found
        openai.OpenAIError: If any OpenAI API requests fail
        pymongo.errors.PyMongoError: If database operations fail
    """
    logger.info(f"Getting descriptions for {len(card_names)} cards")
    
    descriptions = {}
    cards_to_generate = []
    
    # Try to load from database first
    with pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where()) as db:
        # Get the cards collection
        cards = db.mtgai.cards
        
        # Find all existing cards in one query
        existing_cards = list(cards.find({"name": {"$in": card_names}}))
        existing_dict = {card["name"]: card["description"] for card in existing_cards}
        
        logger.debug(f"Found {len(existing_dict)} existing descriptions in database")
        
        # Add existing descriptions and collect missing cards
        for name in card_names:
            if name in existing_dict:
                descriptions[name] = existing_dict[name]
            else:
                cards_to_generate.append(name)
        
        # Generate missing descriptions
        if cards_to_generate:
            logger.debug(f"Generating {len(cards_to_generate)} new descriptions")
            new_descriptions = []
            for name in cards_to_generate:
                card_json = name_to_json.get(name) or get_card_json(name)
                description = generate_card_description(card_json)
                descriptions[name] = description
                new_descriptions.append({
                    "name": name,
                    "description": description
                })
            
            # Store new descriptions in bulk
            if new_descriptions:
                logger.debug(f"Storing {len(new_descriptions)} new descriptions in database")
                cards.insert_many(new_descriptions)
    
    return descriptions

def extract_card_names(decklist_text):
    """
    Extract just the card names from a decklist text that may contain additional information.
    Uses GPT to parse and clean the decklist, removing set codes, collector numbers,
    sideboard/commander labels, quantities and other non-name information.
    
    Args:
        decklist_text (str): Raw decklist text that may contain extra information
        
    Returns:
        list: List of cleaned card names extracted from the decklist
        
    Raises:
        openai.OpenAIError: If the OpenAI API request fails
    """
    logger.info("Extracting card names from decklist")
    system_prompt = """You are an automatic system that extracts card names from Magic: The Gathering decklists.
    You understand common Magic deck formats.
    Given a decklist that may contain extraneous information, extract just the name of each card.
    Return the result as a JSON object with a single field, "card_names", which is an array of strings, where each string is a card name.
    """
    
    logger.debug("Making OpenAI API call to parse decklist")
    response = openai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": decklist_text}
        ],
        temperature=0.1,
        response_format={'type': 'json_object'}
    )
    card_names = json.loads(response.choices[0].message.content)["card_names"]
    logger.debug(f"Extracted {len(card_names)} card names from decklist")
    return card_names

def evaluate_potential_addition(strategy, card_description):
    """
    Evaluates a potential card addition by considering the deck's strategy and the card's description.
    
    Args:
        strategy (str): The deck's strategy
        card_description (str): The card's description
        
    Returns:
        int: A score from 1 to 100 indicating the card's potential usefulness to the deck
    """
    system_prompt = """You are an expert Magic: The Gathering deck builder and advisor.
    You will be given a deck's strategy and a card's description.
    Read the deck's strategy and the card's description carefully.
    Your task is to rate the card's potential usefulness to the deck, on a scale of 1 (worst) to 100 (best).
    Output your score as a JSON object with a single field, "score".
    Example output:
    {"score": 42}
    """
    logger.debug(f"Evaluating potential addition: {card_description.splitlines()[0]}")
    response = openai.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Strategy:\n{strategy}\n\nCard description:\n{card_description}"}],
        temperature=0.1,
        response_format={'type': 'json_object'}
    )
    return json.loads(response.choices[0].message.content)["score"]

def get_potential_additions(current_deck_prompt, current_deck_cards):
    """
    Gets potential additions to a decklist by asking GPT to generate Scryfall search queries for cards that would be good additions, executing those queries against the Scryfall API, and enriching the results with card descriptions.

    Args:
        current_deck_prompt (str): Information about the current deck, including decklist and any additional context

    Returns:
        list: Descriptions of potential card additions. Empty list if no results or error occurs.

    Raises:
        openai.OpenAIError: If the OpenAI API request fails
    """
    system_prompt = """You are an expert Magic: The Gathering deck builder and advisor.
    You will be given information about a deck.
    Your task is to summarize the deck's strategy what kinds of cards might make for good additions, then generate Scryfall search queries to find those cards.
    Your task is NOT to make final decisions about which cards to add, so generate queries to find a range of options that would fill different niches in the deck's strategy.
    Ensure that each query is restricted to legal cards, considering both legality in the format (using the query parameter `f:[format]`) and other restrictions such as color identity (`id<=[color identity]`).
    Be sure to consider any additional information provided, and how it should affect both your description of the deck's strategy and the search queries.
    Your output should be a JSON object with two fields:
    - strategy: A summary of the deck's strategy and what kinds of cards might make for good additions
    - queries: A list of strings where each string is a Scryfall search query
    """
    logger.info("Generating potential additions to decklist")
    
    response = openai.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": current_deck_prompt}],
        temperature=0.1,
        response_format={'type': 'json_object'}
    )
    response_json = json.loads(response.choices[0].message.content)
    strategy = response_json["strategy"]
    queries = response_json["queries"]
    cards = {}
    for query in queries:
        try:
            # Search Scryfall with the query
            logger.debug(f"Running Scryfall query: {query}")
            search_results = requests.get(f"https://api.scryfall.com/cards/search?q={query}")
            if search_results.status_code == 404:
                logger.debug(f"No cards found matching: {query}")
                continue
            search_results.raise_for_status()
            data = search_results.json()
            logger.debug(f"Found {data['total_cards']} matching cards")
            cards.update({card["name"]: card for card in data["data"]})
        except Exception as e:
            logger.error(f"Error running query '{query}': {str(e)}")
    for name in current_deck_cards:
        if name in cards:
            del cards[name]
    descriptions_dict = get_card_descriptions_dict(list(cards.keys()), cards)
    # Sort cards by relevance using evaluate_potential_addition
    sorted_cards = []
    for card_name, description in descriptions_dict.items():
        relevance = evaluate_potential_addition(strategy, description)
        sorted_cards.append((card_name, relevance))
    sorted_cards.sort(key=lambda x: x[1], reverse=True)
    
    # Build final descriptions string with sorted cards
    descriptions = []
    for card_name, relevance in sorted_cards:
        if len(descriptions) > 150 or relevance < 50:
            break
        if card_name in descriptions_dict:
            descriptions.append(descriptions_dict[card_name])
    return descriptions

def get_deck_advice(decklist_text, mode="cheaper", format=None, additional_info=None):
    """
    Gets AI advice on how to improve a decklist by first enriching it with card descriptions
    and then asking for recommendations.
    
    Args:
        decklist_text (str): Raw decklist text containing card names and other info
        
    Returns:
        str: AI advice on improving the deck
        
    Raises:
        openai.OpenAIError: If any OpenAI API requests fail
        requests.exceptions.RequestException: If any Scryfall API requests fail
        ValueError: If no cards are found
    """
    modes = ["cheaper", "better"]
    if mode not in modes:
        raise ValueError(f"Invalid mode: {mode}. Please choose from: {modes}")
    
    logger.info("Getting deck improvement advice")
    # First get descriptions for all cards
    decklist_cards = extract_card_names(decklist_text)
    card_descriptions = get_card_descriptions_dict(decklist_cards)
    card_separator = "\n-----\n"
        
    system_prompt = f"""You are an expert Magic: The Gathering deck builder and advisor.
    You will be given a decklist, along with descriptions of the cards in the deck.
    Read the decklist and card descriptions carefully, noting card quantities when relevant and considering the reason each card is in the deck.
    Your task is to analyze the deck's strategy and provide specific advice on how to improve it.
    Consider aspects like mana curve, overall gameplan, synergies between cards, and potential weaknesses.
    Unless instrtucted otherwise, aim to balance suggestions for cutting and adding cards so that the size of the deck doesn't change.
    Provide reasoning for your suggestions so that players can learn from your advice.
    """
    
    user_prompt = f"""Here are the descriptions of the cards in the deck:
    {card_separator.join([card_descriptions[name] for name in decklist_cards])}
    
    
    Here is the decklist:
    {decklist_text}"""
    
    if format:
        user_prompt += f"\n\nThe decklist is for the {format} format. Consider the rules of {format} when evaluating the deck, and only suggest adding cards that are legal in {format}. Assume that the current decklist is already legal in {format}."
        if format.lower() == "brawl":
            user_prompt += "\nRemember the rules of Brawl (formerly known as Historic Brawl): 2 players, 100-card singleton decks with commanders, restricted to commander's color identity, 25 starting life, using cards from Magic: The Gathering Arena."
    
    if additional_info:
        user_prompt += f"\n\nHere is additional information about the deck:\n{additional_info}"
    
    potential_additions = get_potential_additions(user_prompt, decklist_cards)
    if potential_additions:
        user_prompt += f"\n\nHere are some cards that might be good additions to the deck:\n{card_separator.join(potential_additions)}"
    
    logger.debug("Making OpenAI API call for deck advice")
    response = openai.chat.completions.create(
        model="deepseek-reasoner" if mode == "better" else "deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    logger.debug("Successfully received deck advice from OpenAI")
    return response.choices[0].message.content

if __name__ == "__main__":
    logger.info('Loading decklist')
    teferi_decklist = open("Teferi Deck.txt", "r").read()
    logger.info("Starting deck analysis")
    print(get_deck_advice(teferi_decklist, format="Brawl", additional_info="This deck may not contain creatures."))
