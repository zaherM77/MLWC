from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/"
)
DATASETS = ("results", "goalscorers", "shootouts")

ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 65.0

ELO_K_WEIGHTS = {
    "world_cup": 60.0,         
    "continental_final": 50.0,  
    "qualifier": 40.0,          
    "other_competitive": 30.0,  
    "friendly": 20.0,           
}

ELO_CURRENT_PATH = MODELS_DIR / "elo_current.json"

N_SIMULATIONS = 10_000
RANDOM_SEED = 42

EXTRA_TIME_SCALE = 30.0 / 90.0 
SHOOTOUT_STRENGTH_WEIGHT = 0.5


WORLD_CUP_GROUP_NAMES = list("ABCDEFGHIJKL")

WORLD_CUP_HOSTS = ["United States", "Canada", "Mexico"]

WORLD_CUP_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ROUND_OF_32 = [
    ("1A", "3T1"),
    ("1B", "3T2"),
    ("1C", "3T3"),
    ("1D", "3T4"),
    ("1E", "3T5"),
    ("1F", "3T6"),
    ("1G", "3T7"),
    ("1H", "3T8"),
    ("1I", "2A"),
    ("1J", "2B"),
    ("1K", "2C"),
    ("1L", "2D"),
    ("2E", "2F"),
    ("2G", "2H"),
    ("2I", "2J"),
    ("2K", "2L"),
]

THIRD_SLOT_OPPONENT_GROUP = {
    "3T1": "A", "3T2": "B", "3T3": "C", "3T4": "D",
    "3T5": "E", "3T6": "F", "3T7": "G", "3T8": "H",
}
THIRD_PLACE_ALLOCATION: dict[frozenset, dict[str, str]] = {}

TOURNAMENT_MC_PATH = DATA_DIR / "tournament_mc.json"


ANALYTICS_PATH = DATA_DIR / "analytics.json"

DEFAULT_ADMIN_TOKEN = "change-me-admin"
