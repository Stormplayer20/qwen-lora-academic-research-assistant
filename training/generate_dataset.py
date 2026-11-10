import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path("training/dataset/train.jsonl")


# 50 different academic topic families.
# Each topic has:
# (informal topic description, ideal academic search keywords)
TOPICS = [
    (
        "bias in automated recruitment",
        "algorithmic bias automated hiring recruitment employment discrimination fairness machine learning"
    ),
    (
        "sleep and learning in university students",
        "sleep quality sleep duration academic performance university students cognition memory learning"
    ),
    (
        "social media and concentration",
        "social media attention concentration cognitive performance digital media sustained attention"
    ),
    (
        "student athlete wellbeing",
        "student athletes mental health psychological wellbeing stress anxiety depression collegiate sports"
    ),
    (
        "climate policy effectiveness",
        "climate change environmental policy mitigation adaptation regulation policy effectiveness"
    ),
    (
        "remote work productivity",
        "remote work telecommuting employee productivity workplace performance organizational behavior"
    ),
    (
        "teen social media use",
        "adolescents social media psychological wellbeing anxiety depression digital media"
    ),
    (
        "electric vehicle adoption",
        "electric vehicles consumer adoption transportation sustainability charging infrastructure"
    ),
    (
        "renewable energy storage",
        "renewable energy battery storage energy systems grid integration sustainability"
    ),
    (
        "machine learning healthcare diagnosis",
        "machine learning clinical diagnosis healthcare artificial intelligence predictive models"
    ),
    (
        "cybersecurity awareness",
        "cybersecurity awareness human factors phishing information security behavior"
    ),
    (
        "online learning effectiveness",
        "online education distance learning student outcomes academic achievement digital learning"
    ),
    (
        "exercise and psychological health",
        "physical activity exercise mental health depression anxiety psychological wellbeing"
    ),
    (
        "food insecurity university students",
        "food insecurity university students academic performance nutrition socioeconomic factors"
    ),
    (
        "AI generated misinformation",
        "generative artificial intelligence misinformation disinformation deepfakes information credibility"
    ),
    (
        "privacy in artificial intelligence",
        "artificial intelligence data privacy information security surveillance ethics"
    ),
    (
        "obesity risk factors",
        "obesity risk factors nutrition physical activity socioeconomic determinants public health"
    ),
    (
        "vaccination hesitancy",
        "vaccine hesitancy vaccination attitudes public health misinformation risk perception"
    ),
    (
        "urban green spaces",
        "urban green space public health psychological wellbeing environmental quality cities"
    ),
    (
        "large language model accuracy",
        "large language models evaluation factual accuracy hallucination reliability benchmarking"
    ),
    (
        "AI in education",
        "artificial intelligence education learning outcomes personalized learning academic integrity"
    ),
    (
        "smartphone addiction",
        "smartphone use problematic smartphone behavior addiction psychological wellbeing attention"
    ),
    (
        "workplace burnout",
        "employee burnout occupational stress job satisfaction workplace wellbeing organizational psychology"
    ),
    (
        "income inequality and health",
        "income inequality socioeconomic status health outcomes social determinants public health"
    ),
    (
        "facial recognition bias",
        "facial recognition demographic bias algorithmic fairness computer vision discrimination"
    ),
    (
        "autonomous vehicle safety",
        "autonomous vehicles road safety collision risk artificial intelligence transportation"
    ),
    (
        "video games and aggression",
        "video games aggression violent media behavior adolescents psychology"
    ),
    (
        "music and memory",
        "music memory cognition learning recall neuroscience"
    ),
    (
        "air pollution and health",
        "air pollution respiratory health cardiovascular disease environmental exposure public health"
    ),
    (
        "microplastics in drinking water",
        "microplastics drinking water environmental contamination human exposure health effects"
    ),
    (
        "social isolation elderly adults",
        "social isolation loneliness older adults mental health cognitive decline wellbeing"
    ),
    (
        "childhood obesity prevention",
        "childhood obesity prevention nutrition physical activity public health interventions"
    ),
    (
        "artificial intelligence creativity",
        "artificial intelligence creativity generative models human creativity creative work"
    ),
    (
        "automation and employment",
        "automation employment labor market artificial intelligence job displacement workforce"
    ),
    (
        "gender pay gap",
        "gender wage gap employment discrimination labor economics workplace inequality"
    ),
    (
        "college student debt",
        "student debt higher education financial stress socioeconomic outcomes college students"
    ),
    (
        "economic effects of immigration",
        "immigration economic outcomes labor markets wages employment public policy"
    ),
    (
        "minimum wage employment",
        "minimum wage employment labor economics wages job growth policy"
    ),
    (
        "renewable energy policy",
        "renewable energy policy clean energy regulation sustainability climate mitigation"
    ),
    (
        "water scarcity",
        "water scarcity climate change resource management drought sustainability"
    ),
    (
        "deforestation biodiversity",
        "deforestation biodiversity habitat loss ecosystem conservation environmental change"
    ),
    (
        "mental health telemedicine",
        "telemedicine mental health digital healthcare psychotherapy access treatment outcomes"
    ),
    (
        "wearable health devices",
        "wearable devices digital health monitoring sensors healthcare patient outcomes"
    ),
    (
        "antibiotic resistance",
        "antimicrobial resistance antibiotics bacterial infections public health treatment"
    ),
    (
        "gene editing ethics",
        "CRISPR gene editing bioethics genetic engineering human genome ethics"
    ),
    (
        "robotics in manufacturing",
        "industrial robotics manufacturing automation productivity workforce safety"
    ),
    (
        "blockchain financial systems",
        "blockchain financial technology distributed ledger banking security transactions"
    ),
    (
        "cloud computing security",
        "cloud computing cybersecurity data security privacy distributed systems"
    ),
    (
        "online misinformation elections",
        "online misinformation elections political communication social media disinformation"
    ),
    (
        "artificial intelligence law",
        "artificial intelligence law regulation accountability governance ethics"
    ),
]


# General prompts should produce only the core keyword query.
GENERAL_TEMPLATES = [
    "{topic}",
    "research about {topic}",
    "papers on {topic}",
    "studies about {topic}",
    "what does research say about {topic}",
    "I need academic sources about {topic}",
]


# These prompts explicitly ask for a particular research intent.
# Only then do we add the corresponding modifier to the output.
SPECIAL_TEMPLATES = [
    (
        "recent studies on {topic}",
        "recent research"
    ),
    (
        "systematic reviews about {topic}",
        "systematic review"
    ),
    (
        "foundational research on {topic}",
        "foundational research"
    ),
    (
        "research gaps about {topic}",
        "research gaps"
    ),
]


def main():
    examples = []

    for topic, keywords in TOPICS:

        # 6 normal examples
        for template in GENERAL_TEMPLATES:
            instruction = template.format(topic=topic)

            examples.append({
                "instruction": instruction,
                "output": keywords,
            })

        # 4 intent-specific examples
        for template, modifier in SPECIAL_TEMPLATES:
            instruction = template.format(topic=topic)

            examples.append({
                "instruction": instruction,
                "output": f"{keywords} {modifier}",
            })

    # 50 topics x 10 examples = 500 examples
    random.shuffle(examples)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"Created {len(examples)} training examples.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()