from knowledge_storm.lm import LitellmModel
from lotus.models import LM
import os
from lotus.web_search import WebSearchCorpus, web_search
import dspy

def lotus_to_storm_lm(lm: LM) -> LitellmModel:
    return LitellmModel(
        model=lm.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_type="chat",
        **lm.kwargs
    )
    

class LotusRM(dspy.Retrieve):
    def __init__(self, corpus: list[WebSearchCorpus], k: int, cols: list[str] | None = None, sort_by_date=False):
        super().__init__(k=k)
        self.corpus = corpus
        self.k = k
        self.cols = cols
        self.sort_by_date = sort_by_date
        self.usage = 0

    def get_usage_and_reset(self):
        return {"LotusRM": self.usage}
    
    def forward(self, query_or_queries: str | list[str], exclude_urls: list[str] = []):
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            for corpus in self.corpus:
                df = web_search(corpus, query, self.k, self.cols, self.sort_by_date)
                if len(df) == 0:
                    print(f"No results found for query: {query}")
                    continue
                df["query"] = query
                if corpus == WebSearchCorpus.ARXIV:
                    df.rename(
                        columns={"abstract": "snippet", "link": "url", "published": "date"},
                        inplace=True,
                    )
                    df["date"] = df["date"].astype(str)
                elif (
                    corpus == WebSearchCorpus.GOOGLE or corpus == WebSearchCorpus.GOOGLE_SCHOLAR
                ):
                    df.rename(columns={"link": "url"}, inplace=True)
                elif corpus == WebSearchCorpus.BING:
                    df.rename(columns={"name": "title"}, inplace=True)
                elif corpus == WebSearchCorpus.TAVILY:
                    df.rename(columns={"content": "snippet"}, inplace=True)
        
                for _, row in df.iterrows():
                    if row["url"] not in exclude_urls:
                        result = {
                            "url": row["url"],
                            "title": row["title"],
                            "description": row["snippet"],
                            "snippets": [row["snippet"]],
                        }
                        collected_results.append(result)

        return collected_results
