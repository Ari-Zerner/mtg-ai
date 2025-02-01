import aiohttp
import asyncio
from openai import AsyncOpenAI, BadRequestError
from dotenv import load_dotenv
import os
import pymongo
import certifi
import logging
import re
from bs4 import BeautifulSoup

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
CHEAP_MODEL = env_var("CHEAP_MODEL")
GOOD_MODEL = env_var("GOOD_MODEL")
MONGO_URI = env_var("MONGO_URI")
MAX_CARDS_PER_QUERY = int(env_var("MAX_CARDS_PER_QUERY"))

openai_client = AsyncOpenAI(api_key=env_var("API_KEY"), base_url=env_var("API_BASE_URL"))

async def call_ai(model, dev_prompt, user_prompt):
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except BadRequestError as e:
        if e.code == "unsupported_value" and e.param == "messages[0].role":
            # Fall back to "user" role instead of "developer" for models with unusual API (e.g. o1-mini)
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": dev_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            raise
    return response.choices[0].message.content


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
    dev_prompt = """You are an automatic system that extracts card names from Magic: The Gathering decklists.
    You understand common Magic deck formats.
    Given a decklist that may contain extraneous information, extract just the name of each card.
    Return the result as a list of card names, one per line.
    """
    
    logger.debug("Making OpenAI API call to parse decklist")
    response = await call_ai(CHEAP_MODEL, dev_prompt, decklist_text)
    card_names = [name.strip() for name in response.splitlines() if name.strip()]
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
    dev_prompt = """You are an expert Magic: The Gathering deck builder and advisor.
    You will be given a deck's strategy and a card's description.
    Read the deck's strategy and the card's description carefully.
    Your task is to rate the card's potential usefulness to the deck, on a scale of 1 (worst) to 100 (best).
    Consider both the card's overall strength and how well it synergizes with the deck's strategy.
    Output your score as an integer with no other text.
    """
    try:
        logger.debug(f"Evaluating potential addition: {card_description.splitlines()[0]}")
        user_prompt = f"<strategy>\n{strategy}\n</strategy>\n<card description>\n{card_description}\n</card description>"   
        response = await call_ai(CHEAP_MODEL, dev_prompt, user_prompt)
        return int(response)
    except Exception as e:
        logger.error(f"Error evaluating potential addition {card_description.splitlines()[0]}: {e}")
        return 0

FORMAT_LIST = None
async def get_format_list():
    global FORMAT_LIST
    if FORMAT_LIST:
        return FORMAT_LIST
    
    logger.debug("Fetching format list from Scryfall")
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            async with session.get("https://api.scryfall.com/cards/named?exact=Island") as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch format list: {response.status}")
                    return []
                data = await response.json()
                FORMAT_LIST = sorted(data['legalities'].keys())
                return FORMAT_LIST
    except Exception as e:
        logger.error(f"Error fetching format list: {e}")
        return []

SCRYFALL_SYNTAX_REFERENCE = None
async def get_scryfall_syntax_reference():
    """
    Downloads and extracts the Scryfall syntax reference from their documentation page.
    
    Returns:
        str: Concatenated contents of all reference-block divs from the syntax page
    """
    global SCRYFALL_SYNTAX_REFERENCE
    if SCRYFALL_SYNTAX_REFERENCE:
        return SCRYFALL_SYNTAX_REFERENCE
    
    logger.debug("Downloading Scryfall syntax reference")
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            async with session.get("https://scryfall.com/docs/syntax") as response:
                if response.status != 200:
                    logger.error(f"Failed to download Scryfall syntax reference: {response.status}")
                    return ""
                html = await response.text()
                
        # Parse HTML and extract reference blocks
        soup = BeautifulSoup(html, 'html.parser')
        reference_blocks = soup.find_all('div', class_='reference-block')
        logger.debug(f"Downloaded {len(reference_blocks)} reference blocks")
        reference_text = "<div>\n" + "\n".join(block.get_text(strip=True) for block in reference_blocks) + "\n</div>"
        SCRYFALL_SYNTAX_REFERENCE = reference_text
        return reference_text
    except Exception as e:
        logger.error(f"Error downloading Scryfall syntax reference: {e}")
        return ""

async def get_potential_additions(current_deck_prompt, current_deck_cards, format=None):
    """
    Gets potential additions to a decklist by asking GPT to generate Scryfall search queries for cards that would be good additions, executing those queries against the Scryfall API, and enriching the results with card descriptions.

    Args:
        current_deck_prompt (str): Information about the current deck, including decklist and any additional context
        format (str): The format of the deck, if known
    Returns:
        list: Descriptions of potential card additions. Empty list if no results or error occurs.

    Raises:
        openai.OpenAIError: If the OpenAI API request fails
    """
    syntax_reference = await get_scryfall_syntax_reference()
    dev_prompt = f"""You are an expert Magic: The Gathering deck builder and advisor.
    You will be given information about a deck.
    Your task is to summarize the deck's strategy what kinds of cards might make for good additions, then generate Scryfall search queries to find those cards.
    Your task is NOT to make final decisions about which cards to add, so generate queries to find a range of options that would fill different niches in the deck's strategy.
    Ensure that each query is restricted to legal cards, considering restrictions such as color identity (`id<=[color identity]`).
    Be sure to consider any additional information provided, and how it should affect both your description of the deck's strategy and the search queries.
    Try to keep your queries specific; only the first {MAX_CARDS_PER_QUERY} cards returned by each query will be considered.
    
    Your output should be delineated with XML tags as follows:
    <strategy>
    [summary of the deck's strategy and what kinds of cards might make for good additions]
    </strategy>
    <queries>
    <query>[first Scryfall search query]</query>
    <query>[second Scryfall search query]</query>
    ...
    </queries>
    
    You can use the following syntax reference to help you generate queries:
    <scryfall-syntax-reference>
    {syntax_reference}
    </scryfall-syntax-reference>
    """
    logger.info("Generating potential additions to decklist")
    
    response = await call_ai(GOOD_MODEL, dev_prompt, current_deck_prompt)
    logger.debug(f"Received response: {response}")
    try:
        strategy, queries_block = re.match(r"<strategy>(.+)</strategy>\s*<queries>(.+)</queries>", response, re.DOTALL).groups()
        queries = []
        for query_line in queries_block.splitlines():
            try:
                query_match = re.match(r"<query>(.+)</query>", query_line.strip())
                if query_match:
                    queries.append(query_match.group(1) + (f" f:{format}" if format else ""))
            except Exception as e:
                logger.error(f"Error parsing query line: {query_line}")
                continue
    except:
        logger.error(f"Improperly formatted strategy and queries:\n{response}")
        return []
    
    cards = {}
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        tasks = []
        for query_line in queries:
            tasks.append(asyncio.create_task(fetch_scryfall_search(session, query_line)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for query_line, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Error running query '{query_line}': {str(result)}")
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
    all_cards = []
    url = f"https://api.scryfall.com/cards/search?q={query}"
    while url and len(all_cards) < MAX_CARDS_PER_QUERY:
        async with session.get(url) as response:
            if response.status == 404:
                logger.debug(f"No cards found matching query: {query}")
                return None
            response.raise_for_status()
            data = await response.json()
            all_cards.extend(data["data"])
            url = data.get("next_page")
    logger.debug(f"Found {len(all_cards)} cards matching query: {query}")
    return all_cards[:MAX_CARDS_PER_QUERY]

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
        
    dev_prompt = f"""You are an expert Magic: The Gathering deck builder and advisor.
    You will be given a decklist, along with descriptions of the cards in the deck.
    Read the decklist and card descriptions carefully, noting card quantities when relevant and considering the reason each card is in the deck.
    Your task is to analyze the deck's strategy and provide specific advice on how to improve it.
    Consider aspects like mana curve, overall gameplan, synergies between cards, and potential weaknesses.
    Unless instrtucted otherwise, aim to balance suggestions for cutting and adding cards so that the size of the deck doesn't change.
    Provide reasoning for your suggestions so that players can learn from your advice.
    Make sure your output is properly formatted markdown.
    """
    
    decklist_descriptions_text = '\n'.join([f"<card>\n{card_descriptions[name]}\n</card>" for name in decklist_cards])
    user_prompt = f"""<decklist>
    {decklist_text}
    </decklist>
    
    <decklist-card-descriptions>
    {decklist_descriptions_text}
    </decklist-card-descriptions>
    """
    
    if format:
        format_info = f"The decklist is for the {format} format. Consider the rules of {format} when evaluating the deck. Assume that the current decklist is already legal in {format}."
        if format.lower() == "brawl":
            format_info += "\nRemember the rules of Brawl (formerly known as Historic Brawl): 2 players, 100-card singleton decks with commanders, restricted to commander's color identity, 25 starting life, using cards from Magic: The Gathering Arena."
        user_prompt += f"\n\n<format-info>\n{format_info}\n</format-info>"
    
    if additional_info:
        user_prompt += f"\n\n<additional-info>\n{additional_info}\n</additional-info>"
    
    potential_additions = await get_potential_additions(user_prompt, decklist_cards)
    if potential_additions:
        addition_descriptions_text = '\n'.join([f"<card>\n{card}\n</card>" for card in potential_additions])
        user_prompt += f"\n\n<potential-additions>\n{addition_descriptions_text}\n</potential-additions>"
    
    logger.debug("Making OpenAI API call for deck advice")
    response = await call_ai(GOOD_MODEL, dev_prompt, user_prompt)
    logger.debug("Successfully received deck advice from OpenAI")
    return response