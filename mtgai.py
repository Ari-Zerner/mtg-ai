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
OPENAI_API_KEY = env_var("OPENAI_API_KEY")
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
    url = f"https://api.scryfall.com/cards/search?q={card_name}"
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    if data["total_cards"] == 0:
        logger.error(f"No cards found matching: {card_name}")
        raise ValueError(f"No cards found matching '{card_name}'")
    
    logger.debug(f"Found {data['total_cards']} matching cards")    
    return data["data"][0]

openai = OpenAI(api_key=OPENAI_API_KEY)

def generate_card_description(card_name):
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
    logger.info(f"Generating description for card: {card_name}")
    
    # Get card data from Scryfall
    card_data = get_card_json(card_name)
        
    # Construct prompt for GPT
    system_prompt = """You are an automatic system that parses JSON representations of Magic: The Gathering cards.
    You return plain text detailing all mechanics of the card, and no other information.
    You are familiar with concise representations of card mechanics such as mana symbols.
    Your knowledge of the rules enables you to understand which information is relevant based on the card.
    You don't interpret the card, you just put the information in plain text.
    You are as concise as possible and don't need to use full sentences or proper grammar.
    You always provide all functional information about the card, even if this risks including a small amount of irrelevant information.
    
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
    """
    
    logger.debug("Making OpenAI API call for card description")
    # Get description from GPT
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{card_data}"}
        ],
        temperature=0.1
    )
    logger.debug("Successfully received response from OpenAI")
    return response.choices[0].message.content

def get_card_descriptions(card_names):
    """
    Get natural language descriptions of Magic cards' functional aspects,
    first checking MongoDB in bulk and falling back to Scryfall+AI for missing cards.
    
    Args:
        card_names (list): List of card names to describe
        
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
                description = generate_card_description(name)
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
    Return the result as a JSON array of strings, where each string is a card name.
    """
    
    class ResponseFormat(BaseModel):
        card_names: list[str]
    
    logger.debug("Making OpenAI API call to parse decklist")    
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": decklist_text}
        ],
        temperature=0.1,
        response_format=ResponseFormat
    )
    card_names = response.choices[0].message.parsed.card_names
    logger.debug(f"Extracted {len(card_names)} card names from decklist")
    return card_names

def get_all_card_descriptions(decklist_text):
    """
    Gets descriptions for all cards in a decklist and combines them into a single string.
    
    Args:
        decklist_text (str): Raw decklist text containing card names and potentially other info
        
    Returns:
        str: Combined descriptions of all cards, separated by double newlines
        
    Raises:
        openai.OpenAIError: If any OpenAI API requests fail
        requests.exceptions.RequestException: If any Scryfall API requests fail
        ValueError: If no cards are found
    """
    logger.info("Getting descriptions for all cards in decklist")
    # Extract just the card names
    card_names = extract_card_names(decklist_text)
        
    # Get descriptions for all cards
    descriptions_dict = get_card_descriptions(card_names)
    descriptions = [descriptions_dict[name] for name in card_names]
            
    # Combine all descriptions
    logger.debug(f"Combined {len(descriptions)} card descriptions")
    return "\n\n".join(descriptions)

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
    card_descriptions = get_all_card_descriptions(decklist_text)
        
    system_prompt = f"""You are an expert Magic: The Gathering deck builder and advisor.
    You will be given a decklist, along with descriptions of the cards in the deck.
    Read the decklist and card descriptions carefully, noting card quantities when relevant and considering the reason each card is in the deck.
    Your task is to analyze the deck's strategy and provide specific advice on how to improve it.
    Consider aspects like mana curve, overall gameplan, synergies between cards, and potential weaknesses.
    Unless instrtucted otherwise, aim to balance suggestions for cutting and adding cards so that the size of the deck doesn't change.
    Provide reasoning for your suggestions so that players can learn from your advice.
    """
    
    user_prompt = f"""Here are the descriptions of the cards in the deck:
    {card_descriptions}
    
    
    Here is the decklist:
    {decklist_text}"""
    
    if format:
        user_prompt += f"\n\nThe decklist is for the {format} format. Consider the rules of {format} when evaluating the deck, and only suggest adding cards that are legal in {format}. Assume that the current decklist is already legal in {format}."
        if format.lower() == "brawl":
            user_prompt += "\nRemember the rules of Brawl (formerly known as Historic Brawl): 2 players, 100-card singleton decks with commanders, restricted to commander's color identity, 25 starting life, using cards from Magic: The Gathering Arena."
    
    if additional_info:
        user_prompt += f"\n\nHere is additional information about the deck:\n{additional_info}"
    
    logger.debug("Making OpenAI API call for deck advice")
    if mode == "cheaper":
        model = "gpt-4o"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    elif mode == "better":
        model = "o1-preview"
        messages = [{"role": "user", "content": f"{system_prompt}\n\n\n{user_prompt}"}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    logger.debug("Successfully received deck advice from OpenAI")
    return response.choices[0].message.content

if __name__ == "__main__":
    logger.info('Loading decklist')
    teferi_decklist = open("Teferi Deck.txt", "r").read()
    logger.info("Starting deck analysis")
    print(get_deck_advice(teferi_decklist, format="Brawl", additional_info="This deck may not contain creatures."))
