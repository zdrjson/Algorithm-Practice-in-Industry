import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "data" / "results.json"
STATE_PATH = SCRIPT_DIR / "data" / "push_state.json"
DEFAULT_CONFS = ["kdd", "www", "cikm", "recsys", "wsdm", "sigir", "ecir"]
PRIMARY_KEYWORDS = [
    "click-through", "recommend", "taobao", "ctr", "cvr", "conver", "match",
    "search", "rank", "alipay", "kuanshou", "multi-task", "candidate",
    "relevance", "query", "retriev", "personal", "click", "commerce",
    "embedding", "collaborative", "facebook", "sequential", "wechat",
    "tencent", "multi-objective", "ads", "tower", "approximat",
    "instacart", "airbnb", "negative",
]
SECONDARY_KEYWORDS = [
    "bias", "cold", "a/b", "intent", "product", "domain", "feed",
    "large-scale", "interest", "estima", "online", "twitter",
    "machine learning", "stream", "netflix", "linucb", "user", "term",
    "semantic", "explor", "sampl", "listwise", "constrative", "pairwise",
    "bandit", "variation", "session", "uplift", "distil", "item",
    "similar", "behavior", "cascad", "trigger", "transfer", "top-k",
    "top-n", "bid",
]


class ConfDailyError(RuntimeError):
    """An expected configuration or delivery failure."""


class ConfigurationError(ConfDailyError):
    pass


class DeliveryError(ConfDailyError):
    pass


def parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def env_bool(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_results(path=RESULTS_PATH):
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            first_line = file_handle.readline()
            if first_line.startswith("version https://git-lfs.github.com/spec/"):
                raise ConfigurationError(
                    f"{path} 仍是 Git LFS 指针；请先拉取 LFS 文件"
                )
            file_handle.seek(0)
            results = json.load(file_handle)
    except OSError as exc:
        raise ConfigurationError(f"无法读取会议论文数据 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"会议论文数据不是有效 JSON: {path}") from exc

    if not isinstance(results, dict):
        raise ConfigurationError("会议论文数据顶层必须是 JSON 对象")
    return results


def load_state(path=STATE_PATH):
    if not path.exists():
        return {"version": 1, "sent_paper_ids": []}

    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            state = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"无法读取推送状态 {path}: {exc}") from exc

    if not isinstance(state, dict) or state.get("version") != 1:
        raise ConfigurationError("推送状态文件版本无效")
    sent_ids = state.get("sent_paper_ids")
    if not isinstance(sent_ids, list) or not all(isinstance(item, str) for item in sent_ids):
        raise ConfigurationError("推送状态中的 sent_paper_ids 必须是字符串数组")
    return state


def save_state(sent_paper_ids, path=STATE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "sent_paper_ids": sorted(set(sent_paper_ids)),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    temp_path = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(state, file_handle, indent=2, ensure_ascii=False)
            file_handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def match_score(item):
    if not isinstance(item, dict):
        return -1
    title = str(item.get("paper_name") or "").lower()
    score = 0
    for keyword in PRIMARY_KEYWORDS:
        if keyword in title:
            score += 1
    for keyword in SECONDARY_KEYWORDS:
        if keyword in title:
            score += 0.25
    return score


def stable_paper_id(key, paper):
    url = str(paper.get("paper_url") or "").strip()
    title = " ".join(str(paper.get("paper_name") or "").split()).casefold()
    identity = url or title
    if not identity:
        raise ValueError("paper_url 和 paper_name 不能同时为空")
    digest = hashlib.sha256(
        f"{key.casefold()}\0{identity}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def select_candidates(results, sent_paper_ids, limits, confs, start_year, current_year=None):
    selected = []
    sent_ids = set(sent_paper_ids)
    current_year = current_year or dt.datetime.now().year

    for year in range(current_year, start_year - 1, -1):
        for conf in confs:
            key = f"{conf.lower()}{year}"
            papers = results.get(key)
            if not isinstance(papers, list):
                continue

            ranked_papers = sorted(
                enumerate(papers),
                key=lambda entry: (-match_score(entry[1]), entry[0]),
            )
            for paper_index, paper in ranked_papers:
                if len(selected) >= limits:
                    return selected
                if not isinstance(paper, dict):
                    continue
                abstract = paper.get("paper_abstract")
                if not isinstance(abstract, str) or not abstract.strip():
                    continue
                try:
                    paper_id = stable_paper_id(key, paper)
                except ValueError:
                    continue
                if paper_id in sent_ids:
                    continue
                selected.append({
                    "key": key,
                    "paper_index": paper_index,
                    "paper_id": paper_id,
                    "paper": paper,
                })

    return selected


def validate_feishu_urls(urls):
    if not urls:
        raise ConfigurationError("FEISHU_URL 未配置")
    for index, url in enumerate(urls, start=1):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError(f"FEISHU_URL 第 {index} 项不是有效的 HTTP(S) URL")


def runtime_config():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("DEEPSEEK_API_KEY 未配置")
    feishu_urls = parse_csv(os.environ.get("FEISHU_URL", ""))
    validate_feishu_urls(feishu_urls)
    return api_key, feishu_urls


def translate_with_deepseek(texts, api_key, model="deepseek-chat"):
    if not api_key:
        raise ConfigurationError("DEEPSEEK_API_KEY 未配置")
    if not model.strip():
        raise ConfigurationError("DeepSeek 模型名称不能为空")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigurationError("缺少 openai 依赖，无法调用 DeepSeek") from exc

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30,
        max_retries=2,
    )
    system_prompt = {
        "role": "system",
        "content": (
            "你是一位专业的翻译人员，擅长在人工智能领域内进行高质量的英文到中文翻译。"
            "请准确翻译论文摘要，保留专业术语和技术细节，只输出译文。"
        ),
    }
    translations = []
    for index, text in enumerate(texts, start=1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[system_prompt, {"role": "user", "content": text}],
                temperature=0.3,
                stream=False,
            )
            translated = response.choices[0].message.content
        except Exception as exc:
            raise ConfDailyError(f"DeepSeek 翻译第 {index} 篇摘要失败: {exc}") from exc
        if not isinstance(translated, str) or not translated.strip():
            raise ConfDailyError(f"DeepSeek 翻译第 {index} 篇摘要返回空内容")
        translations.append(translated.strip())
    return translations


def get_org_text(paper):
    orgs = set()
    for author in paper.get("authors_detail") or []:
        if isinstance(author, dict) and author.get("org"):
            orgs.add(str(author["org"]).split(",")[0].strip())
    return "; ".join(sorted(org for org in orgs if org)) or "NA"


def get_authors_text(paper):
    authors = paper.get("paper_authors") or []
    if isinstance(authors, str):
        return authors.strip() or "NA"
    return "; ".join(str(author) for author in authors if author) or "NA"


def build_message(key, paper, index, translation, model):
    today = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    title = str(paper.get("paper_name") or "Untitled")
    url = str(paper.get("paper_url") or "").strip()
    title_text = f"[{title}]({url})" if url else title
    summary = str(paper.get("paper_abstract") or "").strip()
    push_title = f"{key.upper()}[{index}]@{today}"
    content = (
        f"**{key.upper()}** · {title_text}\n\n"
        f"**Author:** {get_authors_text(paper)}\n\n"
        f"**ORG:** {get_org_text(paper)}\n\n"
        f"**Translated (Powered by {model}):**\n\n{translation}\n\n"
        f"**Original abstract:**\n\n{summary}"
    )
    return push_title, content


def send_feishu_message(title, content, urls, dry_run=False):
    if dry_run:
        print(f"[DRY_RUN] {title}\n{content}\n")
        return

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": [{"tag": "markdown", "content": content}],
    }
    payload = {"msg_type": "interactive", "card": card}

    for index, url in enumerate(urls, start=1):
        host = urlparse(url).hostname or "unknown-host"
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            response_data = response.json()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            status_text = str(status_code) if status_code is not None else "unknown"
            raise DeliveryError(
                f"飞书推送端点 {index}/{len(urls)} ({host}) HTTP 失败: "
                f"status={status_text}"
            ) from None
        except requests.RequestException as exc:
            raise DeliveryError(
                f"飞书推送端点 {index}/{len(urls)} ({host}) 请求失败: "
                f"type={type(exc).__name__}"
            ) from None
        except ValueError:
            raise DeliveryError(
                f"飞书推送端点 {index}/{len(urls)} ({host}) 返回 JSON 无效"
            ) from None

        if not isinstance(response_data, dict):
            raise DeliveryError(
                f"飞书推送端点 {index}/{len(urls)} ({host}) 返回格式无效"
            )
        business_code = response_data.get("code", response_data.get("StatusCode"))
        if business_code not in (0, "0"):
            raise DeliveryError(
                f"飞书推送端点 {index}/{len(urls)} ({host}) 业务失败: "
                f"code={business_code}"
            )


def run(args):
    if args.limits <= 0:
        raise ConfigurationError("--limits 必须大于 0")
    if args.push_interval < 0:
        raise ConfigurationError("--push-interval 不能小于 0")
    if args.start_year > dt.datetime.now().year:
        raise ConfigurationError("--start-year 不能晚于当前年份")
    if not args.dry_run and not args.deepseek_model.strip():
        raise ConfigurationError("DeepSeek 模型名称不能为空")

    if args.dry_run:
        api_key, feishu_urls = "", []
    else:
        api_key, feishu_urls = runtime_config()

    results = load_results(args.results)
    state = load_state(args.state)
    confs = [conf.lower() for conf in parse_csv(args.confs)]
    if not confs:
        raise ConfigurationError("--confs 至少需要一个会议名称")

    selected = select_candidates(
        results=results,
        sent_paper_ids=state["sent_paper_ids"],
        limits=args.limits,
        confs=confs,
        start_year=args.start_year,
    )
    if not selected:
        print("没有未推送且已包含摘要的会议论文")
        return 0

    if args.dry_run:
        translations = ["[DRY_RUN] 未调用 DeepSeek" for _ in selected]
    else:
        translations = translate_with_deepseek(
            [candidate["paper"]["paper_abstract"] for candidate in selected],
            api_key,
            args.deepseek_model,
        )
        if len(translations) != len(selected):
            raise ConfDailyError("DeepSeek 返回的译文数量与待推送论文数量不一致")

    sent_ids = set(state["sent_paper_ids"])
    for index, (candidate, translation) in enumerate(
        zip(selected, translations), start=1
    ):
        title, content = build_message(
            candidate["key"],
            candidate["paper"],
            index,
            translation,
            args.deepseek_model,
        )
        send_feishu_message(title, content, feishu_urls, args.dry_run)
        if not args.dry_run:
            sent_ids.add(candidate["paper_id"])
            save_state(sent_ids, args.state)
            print(f"已推送并记录第 {index}/{len(selected)} 篇会议论文")
        if not args.dry_run and index < len(selected) and args.push_interval:
            time.sleep(args.push_interval)

    if not args.dry_run:
        print(f"成功推送并记录 {len(selected)} 篇会议论文")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push unsent conference papers with existing abstracts to Feishu."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--limits", type=int, default=int(os.environ.get("LIMITS", "10")))
    parser.add_argument(
        "--push-interval",
        type=float,
        default=float(os.environ.get("PUSH_INTERVAL", "5")),
    )
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument(
        "--confs", default=os.environ.get("CONFS", ",".join(DEFAULT_CONFS))
    )
    parser.add_argument(
        "--deepseek-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    parser.add_argument("--dry-run", action="store_true", default=env_bool("DRY_RUN", False))
    return parser.parse_args()


def main():
    try:
        return run(parse_args())
    except ConfDailyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
