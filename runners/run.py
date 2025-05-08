import os
import dotenv
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.rm import YouRM
from lotus.models import LM
from lotus.web_search import WebSearchCorpus
from runners.utils import lotus_to_storm_lm, LotusRM
import argparse
import os
import logging
dotenv.load_dotenv()

parser = argparse.ArgumentParser(description="Run the full pipeline.")
parser.add_argument("--query", type=str, help="The query to run the pipeline with.", default="What are recent research methods on speculative decoding for LLMs from the past few weeks?")
parser.add_argument("--output_dir", type=str, help="The output directory for the results.", default="storm_wiki_output")
parser.add_argument("--max_conv_turn", type=int, help="Maximum number of conversation turns.", default=2)
parser.add_argument("--max_perspective", type=int, help="Maximum number of perspectives.", default=2)
parser.add_argument("--max_search_queries_per_turn", type=int, help="Maximum number of search queries per turn.", default=2)
parser.add_argument("--disable_perspective", action="store_true", help="Disable perspective if set.", default=True)
parser.add_argument("--search_top_k", type=int, help="Top k search results to consider.", default=15)
parser.add_argument("--retrieve_top_k", type=int, help="Top k results to retrieve.", default=15)
parser.add_argument("--max_thread_num", type=int, help="Maximum number of threads.", default=10)
args = parser.parse_args()

lm = LM(
    # model="hosted_vllm/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # api_base="http://localhost:8001/v1",
    # custom_llm_provider="hosted_vllm",
    # max_tokens=1000,
    # temperature=0,
    # max_ctx_len=128 * 1000,
)
rm = LotusRM(
    corpus=[WebSearchCorpus.ARXIV],
    k=15,
    sort_by_date=True,
)
engine_args = STORMWikiRunnerArguments(
    output_dir=args.output_dir,
    max_conv_turn=args.max_conv_turn,
    max_perspective=args.max_perspective,
    max_search_queries_per_turn=args.max_search_queries_per_turn,
    disable_perspective=args.disable_perspective,
    search_top_k=args.search_top_k,
    retrieve_top_k=args.retrieve_top_k,
    max_thread_num=args.max_thread_num,
)

lm_configs = STORMWikiLMConfigs()
lm_configs.set_conv_simulator_lm(lotus_to_storm_lm(lm))
lm_configs.set_question_asker_lm(lotus_to_storm_lm(lm))
lm_configs.set_outline_gen_lm(lotus_to_storm_lm(lm))
lm_configs.set_article_gen_lm(lotus_to_storm_lm(lm))
lm_configs.set_article_polish_lm(lotus_to_storm_lm(lm))

logger = logging.getLogger("knowledge_storm")
runner = STORMWikiRunner(engine_args, lm_configs, rm, logger) 

runner.run(
    topic=args.query,
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True,
)
runner.post_run()
runner.summary()