# length baased text splitting

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs_text = loader.load()

text = """Cricket is more than just a sport; it is a global phenomenon.
Born in the fields of England, it traveled across the seas.
It found a second home in the hearts of billions in Asia.
The sound of leather hitting willow is a rhythmic melody.
Every match begins with the high-stakes drama of the toss.
The pitch is the stage where heroes are made and broken.
Test cricket remains the ultimate test of a player’s character.
Five days of battle, strategy, and mental endurance define it.
The white clothes and red ball symbolize a timeless tradition.
Then came One Day Internationals, bringing speed and color.
World Cups have created moments that define entire generations.
The 1983 victory changed the landscape of Indian sports forever.
The T20 format revolutionized the game with its explosive energy.
Power-hitting and creative bowling have become the new norms.
The IPL and other leagues turned the sport into a spectacle.
A perfect cover drive is a masterclass in balance and timing.
A fast bowler charging in is a sight of pure athletic power.
Spinners use guile and flight to deceive the best batters.
The wicketkeeper stands as the silent general behind the stumps.
Fielding has evolved into a display of gravity-defying catches.
A century is a milestone of individual brilliance and grit.
A five-wicket haul is the bowler’s ultimate badge of honor.
The "Nervous Nineties" show the intense pressure of the game.
The "Death Overs" bring a frantic, heart-pounding intensity.
Umpires carry the heavy burden of split-second decision-making.
The DRS system has added a layer of technological precision.
Lord’s is known as the "Mecca of Cricket" for its deep history.
The MCG erupts with the roar of nearly a hundred thousand fans.
The Eden Gardens in Kolkata is legendary for its electric atmosphere.
Rivalries like the Ashes bring a unique, historical intensity.
India versus Pakistan is a match that brings nations to a standstill.
Legends like Don Bradman set records that seem untouchable.
Sachin Tendulkar became a symbol of hope for millions.
Shane Warne turned the art of leg-spin into a global magic show.
Viv Richards redefined batting with his sheer swagger and dominance.
The game teaches us that it isn't over until the last ball is bowled.
It is a sport where the underdog can always stun the giant.
Rain delays and "Duckworth-Lewis" are part of the quirky charm.
The tea break is a polite nod to the sport’s gentlemanly roots.
Cricket unites people across borders, languages, and cultures.
It builds a sense of community in every gully and street corner.
From dusty roads to manicured stadiums, the passion is the same.
Young children dream of wearing their national colors with pride.
The spirit of the game emphasizes fair play and mutual respect.
Captaincy is a complex chess match played on a field of grass.
A hat-trick is a rare feat that electrifies the entire crowd.
The evolution from radio commentary to 4K streaming is vast.
Yet, the core essence of the battle between bat and ball remains.
Cricket is a story of resilience, patience, and sudden glory.
To many, it is not just a game, but a way of life."""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)

result = splitter.split_documents(docs_text)

print(result)