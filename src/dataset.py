"""
Synthetic labeled dataset generator for multi-label aspect classification.

Each training example is a short product sentence or phrase.
Label: a binary vector of length 10 (one bit per aspect).

We generate ~60 examples per aspect (600 total) as synthetic training data.
"""
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

ASPECT_LABELS = [
    "Comfort and Ergonomics",
    "Build Quality and Durability",
    "Performance and Functionality",
    "Design and Aesthetics",
    "Assembly and Ease of Use",
    "Value for Money",
    "Customer Support and Warranty",
    "Packaging and Delivery",
    "Safety and Health",
    "Sensory Experience",
]

# Synthetic training sentences per aspect.
# Each sentence is a positive example for that aspect.
ASPECT_SENTENCES = {
    "Comfort and Ergonomics": [
        "The cushioning is very soft and comfortable.",
        "The lumbar support keeps my back relaxed all day.",
        "It fits perfectly and does not feel tight at all.",
        "Very lightweight and easy to carry around.",
        "The padded armrests reduce strain on my elbows.",
        "The memory foam contours perfectly to my body shape.",
        "The grip feels very natural in my hand.",
        "Wearing it for hours causes no discomfort whatsoever.",
        "The ergonomic shape reduces wrist fatigue significantly.",
        "The seat cushion is thick and very supportive.",
        "Perfectly fitted for my foot size and arch.",
        "The breathable mesh keeps me cool during long sessions.",
        "Great neck support and headrest positioning.",
        "The adjustable height makes it suitable for any desk.",
        "Soft inner lining does not irritate the skin at all.",
        "It distributes weight evenly to avoid pressure points.",
        "The ankle support is excellent during athletic activity.",
        "The backrest angle is perfectly calibrated for posture.",
        "Gentle on sensitive skin with no chafing.",
        "Very comfortable insole with good arch support.",
        "The chair reclines smoothly to a relaxed position.",
        "Cushions retain their shape even after months of use.",
        "The harness fits snugly without being restrictive.",
        "Lightweight design reduces fatigue during travel.",
        "The handle is ergonomically shaped for a firm grip.",
        "Does not cause back pain even after 8 hours of sitting.",
        "The footbed molds to the shape of my foot over time.",
        "Great padding on the shoulder straps of the bag.",
        "The shoe provides excellent heel support while running.",
        "The adjustable straps allow a custom comfortable fit.",
    ],
    "Build Quality and Durability": [
        "The stitching is very neat and strong.",
        "The metal frame feels very solid and sturdy.",
        "The plastic does not flex or creak under pressure.",
        "The zippers are smooth and do not jam or break.",
        "The joints are tightly fitted with no wobble.",
        "The material is thick and resists tearing easily.",
        "The screws hold firmly even after repeated use.",
        "The welds on the frame are clean and even.",
        "The hinge mechanism is very solid with no play.",
        "The rubber base prevents sliding and adds stability.",
        "After six months of daily use there is no wear visible.",
        "The outer shell is scratch resistant and hard.",
        "The coating has not chipped or peeled off yet.",
        "The springs inside the cushion retain their bounce.",
        "The thick sole on the shoe shows no signs of wear.",
        "The steel legs of the table have not bent at all.",
        "The buckle snaps securely and has not broken.",
        "The wooden surface is smooth and well lacquered.",
        "The wiring inside is neatly wrapped and insulated.",
        "The velcro strips are still strong after hundreds of uses.",
        "The lid closes tightly with no gaps or leaks.",
        "The handles have not come loose despite heavy loads.",
        "The mesh fabric is tear resistant and holds shape.",
        "The rivets are firmly hammered with no sharp edges.",
        "The bolt holes are precisely drilled and aligned.",
        "The rubber seal on the lid prevents any water ingress.",
        "The leather has not cracked or faded after a year.",
        "The locking mechanism clicks firmly with no play.",
        "The aluminum alloy body has not dented despite drops.",
        "The base plate feels extremely heavy and well made.",
    ],
    "Performance and Functionality": [
        "The motor is very powerful and runs without any lag.",
        "The battery lasts the full day without needing a charge.",
        "The signal is strong and the connection is stable.",
        "The cutting blade stays sharp even after extended use.",
        "The cooling fan keeps the temperature very low.",
        "The processor handles multitasking without slowing down.",
        "The pumping speed is very fast and efficient.",
        "The suction power is excellent and picks up fine dust.",
        "The camera takes very sharp and detailed photos.",
        "The algorithm predicts results accurately every time.",
        "The speaker output is loud and clear at full volume.",
        "The charging speed is much faster than the older model.",
        "The watch tracks my heart rate with high accuracy.",
        "The blender crushes ice in under ten seconds.",
        "The response time of the screen is very fast.",
        "The GPS locks onto the position within seconds.",
        "The automatic transmission shifts smoothly at speed.",
        "The air purifier cleans the room in under an hour.",
        "The router covers the entire apartment with no dead zones.",
        "The mixer handles thick dough without overheating.",
        "The tire grip is excellent even on wet roads.",
        "The torch beam is very bright and reaches far.",
        "The scale gives consistent readings every time.",
        "The laser cuts through the material precisely.",
        "The projection is very clear even in bright daylight.",
        "The pressure washer removes stains in a single pass.",
        "The sensor detects motion from up to ten meters away.",
        "The heater reaches target temperature in two minutes.",
        "The filter removes all impurities from the water.",
        "The autofocus locks instantly on moving subjects.",
    ],
    "Design and Aesthetics": [
        "The color matches the product photos exactly.",
        "The overall look is very modern and sleek.",
        "The matte finish gives it a premium feel.",
        "The logo is embossed cleanly and looks elegant.",
        "The curves and proportions are very well designed.",
        "The chrome accents add a sophisticated touch.",
        "The packaging design itself looks very attractive.",
        "The color scheme is harmonious and pleasing to the eye.",
        "The transparent top panel shows the internals nicely.",
        "The slim profile fits perfectly on a narrow desk.",
        "The design language is consistent across all components.",
        "The LED lighting adds a stylish glow at night.",
        "The wood grain texture looks realistic and premium.",
        "The typeface on the label is crisp and readable.",
        "The stitching pattern on the bag is very decorative.",
        "The product looks much more expensive than it is.",
        "The color is a beautiful deep navy not too bright.",
        "The frosted glass panel looks sophisticated.",
        "The shape is unique and stands out from competitors.",
        "The minimal design without clutter looks very clean.",
        "The watch face has an elegant and readable layout.",
        "The cabinet blends well with the living room furniture.",
        "The textile print is vibrant and does not fade.",
        "The product looks identical to the advertisement images.",
        "The gradient finish on the lid is very eye catching.",
        "The rose gold trim is tasteful and not overdone.",
        "The matte black version looks incredibly premium.",
        "The font choice and sizing on the dial is perfect.",
        "The symmetry of the design is flawless.",
        "The compact dimensions are well proportioned.",
    ],
    "Assembly and Ease of Use": [
        "The instruction manual has clear diagrams.",
        "It took less than twenty minutes to fully assemble.",
        "All the required tools are included in the box.",
        "The buttons and controls are intuitively placed.",
        "The app paired with the device instantly.",
        "The screws fit perfectly into the pre-drilled holes.",
        "No additional tools are required for installation.",
        "The touch controls are highly responsive.",
        "The voice commands work accurately every time.",
        "The on-screen guide made setup very easy.",
        "The parts snapped together without any force.",
        "The cable management options are very well thought out.",
        "The instruction booklet is available in multiple languages.",
        "The single power button controls all main functions.",
        "It works out of the box without any configuration.",
        "The assembly video on the brand website is very helpful.",
        "I assembled the entire unit alone in thirty minutes.",
        "The labels on each component made identification simple.",
        "The plug and play setup requires no driver installation.",
        "The adjustment knobs are easy to reach and turn.",
        "The foldable design makes it very compact for storage.",
        "All bolts are pre-threaded making assembly faster.",
        "The locking tabs click firmly and release easily.",
        "The user interface is clean and easy to navigate.",
        "The daily operation requires only two buttons.",
        "The product starts working immediately when plugged in.",
        "The packaging is designed to guide disassembly logically.",
        "The quick release mechanism works without any tools.",
        "Installing the battery takes under a minute.",
        "The dial makes fine adjustments very intuitive.",
    ],
    "Value for Money": [
        "The quality is exceptional for the price point.",
        "Compared to branded alternatives this is much cheaper.",
        "I did not expect such good performance at this budget.",
        "The price to quality ratio is outstanding.",
        "Totally worth every rupee I paid.",
        "Much better than the expensive models I tried before.",
        "The features offered at this price are unbeatable.",
        "A premium product available at an economy price.",
        "The durability justifies the slightly higher price.",
        "Very affordable without compromising on quality.",
        "I got exactly what was advertised at a fair price.",
        "I would buy this again at the same price.",
        "The cost of ownership over three years is very low.",
        "The replacement parts are affordable and widely available.",
        "The accessories included add tremendous value.",
        "It outperforms products that cost three times as much.",
        "The annual membership fee is justified by the savings.",
        "Free shipping at this price point is a great bonus.",
        "Buying this saved me money compared to hiring a professional.",
        "The long warranty period adds significant value.",
        "The bundle deal made it very cost effective.",
        "The energy saving mode reduces electricity bills.",
        "Cheaper than the competition with better build quality.",
        "No hidden charges or subscription fees after purchase.",
        "The resale value of this brand remains high.",
        "Very competitive pricing compared to the market average.",
        "The discounted sale price made it an incredible deal.",
        "I feel I received more value than I paid for.",
        "The lifetime guarantee makes the upfront cost worthwhile.",
        "The product has paid for itself in savings already.",
    ],
    "Customer Support and Warranty": [
        "The brand replaced my defective unit without any questions.",
        "Customer care responded to my email within an hour.",
        "The warranty covers all manufacturing defects for two years.",
        "The return process was completely hassle free.",
        "The support agent was polite and resolved my issue quickly.",
        "A technician arrived within two days to fix the problem.",
        "The brand honored the warranty without any paperwork.",
        "The online chat support is available around the clock.",
        "The escalation team resolved my complaint in one call.",
        "The company proactively reached out after my negative review.",
        "The extended warranty plan is affordable and comprehensive.",
        "Returning the product for a full refund was very easy.",
        "The brand sent a replacement part by express courier.",
        "The service center staff are knowledgeable and friendly.",
        "My query on social media was answered within minutes.",
        "The warranty registration process takes under a minute.",
        "Post purchase customer care is genuinely helpful.",
        "The brand issued a software update to fix my reported bug.",
        "I received a full refund without needing to return the item.",
        "The technical support team walked me through the fix remotely.",
        "The service engineer performed a thorough check up.",
        "The brand sent a gift card as apology for the delay.",
        "The toll free number connects to a real agent quickly.",
        "My issue was logged and resolved within 24 hours.",
        "The brand offers a no questions asked return policy.",
        "Annual maintenance contract is affordable and reliable.",
        "The online knowledge base answered my question instantly.",
        "The dealer provided on site support without extra charge.",
        "The company takes customer feedback very seriously.",
        "The post sale follow up call was a nice surprise.",
    ],
    "Packaging and Delivery": [
        "The product arrived in perfect condition with no damage.",
        "The packaging used double layered foam for protection.",
        "Delivered one day earlier than the estimated date.",
        "All items listed on the box were present inside.",
        "The unboxing experience felt premium and well thought out.",
        "The outer box has a tamper evident seal.",
        "The inner packaging had separate compartments for each part.",
        "Arrived in a thick corrugated box with bubble wrap.",
        "The delivery executive handled the package with care.",
        "The product was factory sealed and untampered.",
        "The accessories were individually wrapped in plastic.",
        "The gift wrapping option was available and neat.",
        "Delivered to the doorstep with a signature confirmation.",
        "The tracking updates were accurate at every stage.",
        "Packaging was minimal and eco friendly.",
        "The fragile items were marked clearly on the box.",
        "The cardboard is thick enough to withstand drops.",
        "The product was packed tightly with no room to shift.",
        "Received the order within two days of placing it.",
        "The order confirmation and invoice were included in the box.",
        "The box size was appropriately sized for the product.",
        "The packing tape was applied neatly on all seams.",
        "The unboxing video I recorded shows it arrived pristine.",
        "The quick delivery exceeded my expectations entirely.",
        "The serial number sticker on the product matches the box.",
        "Express delivery was worth the small additional fee.",
        "No dents or deformations on the product packaging.",
        "The product is sealed in an airtight bag inside the box.",
        "The brand uses recycled materials for all packaging.",
        "The soft foam padding prevented any scratches on arrival.",
    ],
    "Safety and Health": [
        "The product carries a BIS certification for safety.",
        "The materials are food grade and non toxic.",
        "No sharp edges or protrusions that could cause injury.",
        "The electrical wiring has proper insulation.",
        "The device has an overheating protection circuit.",
        "The smoke alarm triggers within seconds of detection.",
        "The helmet meets ISI safety standards for impact.",
        "The chemicals used are completely biodegradable.",
        "The knee guard provides full protection against impacts.",
        "The harness has passed drop test certification.",
        "The product is free from BPA and phthalates.",
        "The baby toy has rounded edges with no small parts.",
        "The fire retardant coating meets international standards.",
        "The non slip base prevents accidents on wet floors.",
        "The sunscreen formula is dermatologist tested and approved.",
        "The child lock feature prevents accidental activation.",
        "The voltage stabilizer protects against power surges.",
        "The gas regulator has a safety cut off valve.",
        "The water purifier removes all harmful bacteria.",
        "The air quality monitor alerts when pollution spikes.",
        "The fume hood prevents chemical vapors from escaping.",
        "The reflective strips improve visibility at night.",
        "The stroller has a five point harness for infant safety.",
        "The crib meets all safety standards for co sleeping.",
        "The pressure cooker has a double safety valve.",
        "The swimming goggles seal properly to keep water out.",
        "The gloves are puncture resistant and meet EN388.",
        "The product has zero formaldehyde in its composition.",
        "The UV filter on the sunglasses is 100 percent rated.",
        "The car seat passed five star crash test evaluation.",
    ],
    "Sensory Experience": [
        "The flavor is rich and exactly what was advertised.",
        "The fragrance lasts more than twelve hours on fabric.",
        "The scent is a perfect blend of floral and musk notes.",
        "The taste is sweet and not artificial at all.",
        "The sound quality from the speaker is crisp and warm.",
        "The bass is deep and the treble is clean without distortion.",
        "The texture is smooth and melts in the mouth.",
        "The aroma fills the entire room within minutes.",
        "The earphones produce a very immersive spatial sound.",
        "The coffee has a strong roasted flavor with low bitterness.",
        "The moisturizer has a light floral scent that is pleasant.",
        "The sound insulation of the room is excellent.",
        "The velvet fabric feels incredibly soft to the touch.",
        "The taste of the protein powder blends well with milk.",
        "The essential oil has a very authentic lavender aroma.",
        "The noise cancelling blocks out all ambient sounds.",
        "The wine has a complex bouquet with notes of cherry.",
        "The soap lathers very well and leaves skin silky.",
        "The chocolate has a perfectly balanced cocoa intensity.",
        "The guitar produces a warm and resonant tone.",
        "The shampoo has a refreshing mint and eucalyptus scent.",
        "The speaker reproduces live instruments very accurately.",
        "The lotion absorbs quickly and leaves no greasy residue.",
        "The tea has a mild earthy flavor with a smooth finish.",
        "The room freshener releases a consistent gentle scent.",
        "The silk pillowcase feels smooth and cool on the skin.",
        "The pickle has the right balance of spice and sourness.",
        "The vinyl record player produces a warm analog sound.",
        "The perfume has excellent projection and longevity.",
        "The energy drink has a pleasant citrus burst on first sip.",
    ],
}


def build_training_data() -> Tuple[List[str], List[List[int]]]:
    """
    Build training examples from the synthetic sentences.
    Returns:
        texts: list of sentence strings
        labels: list of binary label vectors (length = 10)
    """
    texts, labels = [], []
    for asp_idx, asp_name in enumerate(ASPECT_LABELS):
        sentences = ASPECT_SENTENCES.get(asp_name, [])
        for sentence in sentences:
            label_vec = [0] * len(ASPECT_LABELS)
            label_vec[asp_idx] = 1
            texts.append(sentence)
            labels.append(label_vec)
    return texts, labels


class SyntheticAspectDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic labeled training data.
    Each item is a (text, label_vector) pair.
    """
    def __init__(self):
        self.texts, self.labels = build_training_data()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ─── Legacy classes kept for backwards compatibility ───────────────────────
class DatasetManager:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def load_all_products(self):
        texts, labels = build_training_data()
        records = []
        for t, l in zip(texts, labels):
            records.append({"input_text": t, "aspect_labels": l, "subaspect_labels": []})
        return records

    def get_target_taxonomies(self, records):
        return ASPECT_LABELS, []

    def prepare_splits(self, val_ratio=0.15, test_ratio=0.15):
        texts, labels = build_training_data()
        n = len(texts)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test

        def make_records(ts, ls):
            return [{"input_text": t, "aspect_labels": l, "subaspect_labels": []}
                    for t, l in zip(ts, ls)]

        return (
            make_records(texts[:n_train], labels[:n_train]),
            make_records(texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
            make_records(texts[n_train + n_val:], labels[n_train + n_val:]),
        )


class ProductAspectDataset(Dataset):
    def __init__(self, records, target_aspects, target_subaspects):
        self.records = records
        self.target_aspects = target_aspects

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "input_text": r["input_text"],
            "aspect_target": torch.tensor(r["aspect_labels"], dtype=torch.float32),
            "subaspect_target": torch.zeros(1),
        }
