from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
import asyncio
import csv

load_dotenv()

model = AzureOpenAILLM(
    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model_params={
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
)

embbeding = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

driver.verify_connectivity()

text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

NODES = [
    "Person",
    "Skill",
    "Company",
    "Project",
    "University",
    "Degree"
]


RELATIONSHIP_TYPES = [
    "HAS_SKILL",
    "WORKED_AT",
    "WORKED_ON",
    "STUDIED_AT",
    "HAS_DEGREE",
    "USES"
]

PATTERNS = [
    ("Person", "HAS_SKILL", "Skill"),
    ("Person", "WORKED_AT", "Company"),
    ("Person", "WORKED_ON", "Project"),
    ("Person", "STUDIED_AT", "University"),
    ("Person", "HAS_DEGREE", "Degree"),
    ("Project", "USES", "Skill")
]

kg_builder = SimpleKGPipeline(
    llm=model,
    driver=driver,
    embedder=embbeding,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    from_pdf=True,
    text_splitter=text_splitter,
    schema={
        "node_types": NODES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS
    }
)

data_path = "data"
docs = csv.DictReader(
    open(os.path.join(data_path, "docs.csv"), encoding="utf8", newline="")
)

    
asyncio.run(kg_builder.run_async(file_path="cv.pdf"))
