import json
import aiohttp
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import pymongo
import certifi
import logging
import re

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load environment variables
def env_var(name, default=None):
    value = os.getenv(name)
    if not value:
        if default:
            logger.debug(f"Environment variable {name} not found, using default value: {default}")
            return default
        else:
            logger.error(f"Environment variable {name} not found")
            raise ValueError(f"{name} is not set in the environment variables")
    logger.debug(f"Loaded environment variable: {name}")
    return value

logger.info("Loading environment variables")
DEEPSEEK_API_KEY = env_var("DEEPSEEK_API_KEY")
MONGO_URI = env_var("MONGO_URI")
USE_ONLY_CHEAP_MODEL = env_var("USE_ONLY_CHEAP_MODEL", "false") == "true"

openai = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/")

async def fetch_card_description(session, name):
    """Helper function to fetch a single card description from Scryfall"""
    logger.debug(f"Searching Scryfall for card: {name}")
    url = f"https://api.scryfall.com/cards/named?fuzzy={aiohttp.helpers.quote(name)}&format=text"
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()

async def get_card_descriptions_dict(card_names):
    """
    Get natural language descriptions of Magic cards' functional aspects,
    first checking MongoDB in bulk and falling back to Scryfall+AI for missing cards.
    
    Args:
        card_names (list): List of card names to describe
    Returns:
        dict: Dictionary mapping card names to their descriptions
        
    Raises:
        aiohttp.ClientError: If any Scryfall API requests fail
        ValueError: If no cards are found
        openai.OpenAIError: If any OpenAI API requests fail
        pymongo.errors.PyMongoError: If database operations fail
    """
    logger.info(f"Getting descriptions for {len(card_names)} cards")
    
    descriptions = {}
    cards_to_get = []
    
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
                cards_to_get.append(name)
        
        # Generate missing descriptions
        if cards_to_get:
            logger.debug(f"Getting {len(cards_to_get)} new descriptions")
            new_descriptions = []
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
                tasks = []
                for name in cards_to_get:
                    tasks.append(asyncio.create_task(fetch_card_description(session, name)))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for name, result in zip(cards_to_get, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error searching Scryfall for card '{name}': {str(result)}")
                        continue
                    descriptions[name] = result
                    new_descriptions.append({
                        "name": name,
                        "description": result
                    })
            
            # Store new descriptions in bulk
            if new_descriptions:
                logger.debug(f"Storing {len(new_descriptions)} new descriptions in database")
                cards.insert_many(new_descriptions)
    
    return descriptions

async def extract_card_names(decklist_text):
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
    Return the result as a list of card names, one per line.
    """
    
    logger.debug("Making OpenAI API call to parse decklist")
    response = await openai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": decklist_text}
        ],
        temperature=0.1
    )
    card_names = [name.strip() for name in response.choices[0].message.content.splitlines() if name.strip()]
    logger.debug(f"Extracted {len(card_names)} card names from decklist")
    return card_names

async def evaluate_potential_addition(strategy, card_description):
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
    Consider both the card's overall strength and how well it synergizes with the deck's strategy.
    Output your score as an integer with no other text.
    Example output:
    42
    """
    try:
        logger.debug(f"Evaluating potential addition: {card_description.splitlines()[0]}")
        response = await openai.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Strategy:\n{strategy}\n\nCard description:\n{card_description}"}],
            temperature=0.1
        )
        return int(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error evaluating potential addition {card_description.splitlines()[0]}: {e}")
        return 0

async def get_potential_additions(current_deck_prompt, current_deck_cards):
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
    Your output should contain two sections, separated by a double newline:
    - A summary of the deck's strategy and what kinds of cards might make for good additions
    - A list of Scryfall search queries, one per line
    
    Example output format:
    The deck is a green-based stompy deck with a focus on ramping into large creatures, particularly Dinosaurs, and leveraging +1/+1 counters for additional value. The commander, Ghalta, Primal Hunger, benefits from having high-power creatures on the battlefield to reduce its casting cost. The deck includes a mix of ramp spells, large creatures with trample, and cards that synergize with +1/+1 counters. Good additions would be more ramp spells to ensure casting Ghalta and other large creatures early, additional large creatures with trample or other forms of evasion, and cards that provide card draw or protection to maintain board presence and deal with opposing threats.
    
    Scryfall search queries:
    f:brawl id<=g t:Dinosaur
    f:brawl id<=g pow>=4 tou>=4
    f:brawl id<=g o:"+1/+1 counter"
    f:brawl id<=g o:"add "
    f:brawl id<=g o:"search" o:"library" o:"land"
    """
    logger.info("Generating potential additions to decklist")
    
    response = await openai.chat.completions.create(
        model="deepseek-chat" if USE_ONLY_CHEAP_MODEL else "deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_deck_prompt}
        ]
    )
    response_text = response.choices[0].message.content
    logger.debug(f"Received response: {response_text}")
    try:
        strategy, queries = re.match(r"(.+)\n\n(.+)", response_text, re.DOTALL).groups()
        queries = [query.strip() for query in queries.splitlines() if query.strip()]
    except:
        logger.error(f"Improperly formatted strategy and queries:\n{response_text}")
        return []
    
    cards = {}
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        tasks = []
        for query in queries:
            tasks.append(asyncio.create_task(fetch_scryfall_search(session, query)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Error running query '{query}': {str(result)}")
                continue
            if result:
                cards.update({card["name"]: card for card in result})
    
    for name in current_deck_cards:
        if name in cards:
            del cards[name]
            
    card_names = list(cards.keys())
    descriptions_dict = await get_card_descriptions_dict(card_names)
    
    # Sort cards by relevance using evaluate_potential_addition
    eval_tasks = [asyncio.create_task(evaluate_potential_addition(strategy, descriptions_dict[card_name])) for card_name in card_names]
    relevance_scores = await asyncio.gather(*eval_tasks)
    sorted_cards = sorted(zip(card_names, relevance_scores), key=lambda x: x[1], reverse=True)
    
    # Build final descriptions string with sorted cards
    descriptions = []
    for card_name, relevance in sorted_cards:
        if len(descriptions) > 150 or relevance < 50:
            break
        if card_name in descriptions_dict:
            descriptions.append(descriptions_dict[card_name])
    return descriptions

async def fetch_scryfall_search(session, query):
    """Helper function to search Scryfall with a query"""
    logger.debug(f"Running Scryfall query: {query}")
    async with session.get(f"https://api.scryfall.com/cards/search?q={query}") as response:
        if response.status == 404:
            logger.debug(f"No cards found matching: {query}")
            return None
        response.raise_for_status()
        data = await response.json()
        logger.debug(f"Found {data['total_cards']} matching cards")
        return data["data"]

async def get_deck_advice(decklist_text, format=None, additional_info=None):
    """
    Gets AI advice on how to improve a decklist by first enriching it with card descriptions
    and then asking for recommendations.
    
    Args:
        decklist_text (str): Raw decklist text containing card names and other info
        
    Returns:
        str: AI advice on improving the deck
        
    Raises:
        openai.OpenAIError: If any OpenAI API requests fail
        aiohttp.ClientError: If any Scryfall API requests fail
        ValueError: If no cards are found
    """
    logger.info("Getting deck improvement advice")
    # First get descriptions for all cards
    decklist_cards = await extract_card_names(decklist_text)
    card_descriptions = await get_card_descriptions_dict(decklist_cards)
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
    
    potential_additions = await get_potential_additions(user_prompt, decklist_cards)
    if potential_additions:
        user_prompt += f"\n\nHere are some cards that might be good additions to the deck:\n{card_separator.join(potential_additions)}"
    
    logger.debug("Making OpenAI API call for deck advice")
    response = await openai.chat.completions.create(
        model="deepseek-chat" if USE_ONLY_CHEAP_MODEL else "deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    logger.debug("Successfully received deck advice from OpenAI")
    return response.choices[0].message.content
