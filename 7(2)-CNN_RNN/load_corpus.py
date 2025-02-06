# 구현하세요!

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    corpus = dataset["text"]
    # 빈 문자열 제외
    corpus = [line for line in corpus if line.strip() != ""]
    return corpus