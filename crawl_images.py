"""
Script crawl ảnh lá cải (mustard leaf) từ Bing Images theo gợi ý của bạn.

Ảnh thô sẽ được lưu vào:
  raw_images/
    healthy/
    yellow/
    pest/
    yellow_pest/

Sau đó bạn có thể chạy script prepare_dataset.py để resize + đưa vào dataset/train & dataset/val.
"""

from icrawler.builtin import BingImageCrawler
from pathlib import Path


RAW_ROOT = Path("raw_images")


SEARCH_CONFIG = {
    "healthy": {
        "keywords": [
            "mustard leaf healthy",
            "brassica juncea healthy leaf",
            "mustard greens healthy leaves",
        ],
        "max_num": 300,
    },
    "yellow": {
        "keywords": [
            "mustard leaf yellow disease",
            "mustard leaf chlorosis",
            "brassica juncea yellow leaf",
        ],
        "max_num": 300,
    },
    "pest": {
        "keywords": [
            "mustard leaf pest damage",
            "mustard leaf insect damage",
            "brassica juncea leaf holes",
        ],
        "max_num": 300,
    },
    "yellow_pest": {
        "keywords": [
            "mustard leaf pest and yellow disease",
            "mustard leaf fungal disease spots",
        ],
        "max_num": 300,
    },
}


def crawl_group(group_name: str, keywords: list[str], max_num: int) -> None:
    out_dir = RAW_ROOT / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for kw in keywords:
        print(f"Crawling '{kw}' into {out_dir} ...")
        crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
        # Nhiều search engine giới hạn, nên dùng num per keyword nhỏ hơn
        crawler.crawl(keyword=kw, max_num=max_num)


def main():
    RAW_ROOT.mkdir(exist_ok=True)
    for group, cfg in SEARCH_CONFIG.items():
        crawl_group(group, cfg["keywords"], cfg["max_num"])


if __name__ == "__main__":
    main()

