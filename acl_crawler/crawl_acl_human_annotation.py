''' 
This code is based on and adapted from https://github.com/srhthu/ACL-Anthology-Crawler.
'''

from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
import re
import time

# add conference and time range to crawl
conferences = ["acl", "emnlp", "naacl", "eacl", "aacl", "tacl"] 
years = range(2018, 2027)

# add keywords list to match title or abstract
keywords = ["manual annotation", "human evaluation", "manual evaluation", "manual annotation", 
            "human raters", "human annotator", "human respondent", "human rating", "manual rating", 
            "crowdsourcer", "crowdsourcing", "hil", "human in the loop", "human-in-the-loop",
            "annotator", "labeler", "human expert", "student worker", 
            "amazon mechanical turk", "prolific", "human perception", "human study", "human benchmark", 
            "human dataset", "human performance", "mturk", "expert annotation", "expert evaluation", 
            "crowdworker", "expert opinion", "human judgement", "human judgment", "manually annotated",
            "manually evaluated", "human annotation" ]

# folder to save papers
save_root = "./pdfs"

BASE_URL = "https://aclanthology.org/events/{conf}-{year}/"


# extract html soup
def get_bs_soup_from_url(url):
    html_doc = requests.get(url).text
    return BeautifulSoup(html_doc, 'html.parser')

# return main section with subsections
def get_main_section(html_soup):
    return html_soup.find(id = 'main')

# find subsection titles
def find_title(soup_obj):
    h_tags = soup_obj.find_all(['h2', 'h3', 'h4', 'h5', 'h6'], class_ = 'd-sm-flex')
    for h in h_tags:
        # Find all "a" tag, whose class do not contain "badge"
        for a in h.find_all('a'):
            if 'badge' not in a.get('class', []):
                return a.get_text().strip()
    return "Unknown Section Title"

# get list of subsections
def get_sub_sections(main_section):
    sub_sections = []
    for div in main_section.find_all('div', recursive = False):
        if 'id' in div.attrs:
            sub_section_id = div['id']
            sub_section_title = find_title(div)
            sub_sections.append((sub_section_id, sub_section_title))
    return sub_sections

# get paper list
def find_paper_list(soup, section_id):
    paper_list = []
    section_div = soup.find('div', id=section_id)

    if section_div is None:
        print("Section not found:", section_id)
        return paper_list

    # Only grab actual paper entries
    entries = section_div.find_all(
        "div",
        class_="d-sm-flex align-items-stretch mb-3"
    )

    print("Entries found:", len(entries))

    for entry in entries:

        try:
            # pdf
            pdf_tag = entry.find("a", title="Open PDF")
            if not pdf_tag:
                continue
            pdf_url = pdf_tag["href"]

            # title
            strong = entry.find("strong")
            if not strong:
                continue

            # url        
            paper_a = strong.find("a")
            title = paper_a.get_text(strip=True)
            url = "https://aclanthology.org" + paper_a["href"]

            # abstract
            abstract = ""
            next_div = entry.find_next_sibling("div")
            if next_div and "abstract-collapse" in next_div.get("class", []):
                abstract = next_div.get_text(" ", strip=True)

            paper_list.append({
                "title": title,
                "url": url,
                "pdf_url": pdf_url,
                "abstract": abstract
            })

        except Exception as e:
            continue

    return paper_list

# match keywords in abstract and title
def match_keywords(papers, keywords):
    filtered = []
    keywords = [k.strip().lower() for k in keywords if k.strip()]

    for paper in papers:
        text = (paper.get("title", "") + " " +
                paper.get("abstract", "")).lower()

        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                filtered.append(paper)
                break 

    return filtered

# save papers to system
def save_papers(paper_list, conf_name):
    os.makedirs(conf_name, exist_ok=True)

    for i, paper in tqdm(list(enumerate(paper_list))):

        try:
            r = requests.get(paper["pdf_url"])

            # sanitize filename, otherwise error on windows
            title = paper["title"]

            title = re.sub(r'[\\/*?:"<>|]', "", title)

            title = title.replace("“", "").replace("”", "").replace("’", "")

            title = re.sub(r'\s+', ' ', title).strip()

            max_len = 150
            if len(title) > max_len:
                title = title[:max_len]

            filename = f"{i+1}.{title}.pdf"

            filepath = os.path.join(conf_name, filename)

            with open(filepath, "wb") as f:
                f.write(r.content)

        except Exception as e:
            print("Failed saving:", paper.get("title", "Unknown"))
            print("Error:", e)

# count keyword hits in paper
def count_keyword_hits(papers, keywords):

    counts = {k: 0 for k in keywords}
    keywords = [k.strip().lower() for k in keywords if k.strip()]

    for paper in papers:
        text = (paper.get("title", "") + " " +
                paper.get("abstract", "")).lower()

        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                counts[kw] += 1

    return counts

# main
for conf in conferences:
    for year in years:

        print(f"\nProcessing {conf.upper()} {year}")

        url = BASE_URL.format(conf=conf, year=year)
        soup = get_bs_soup_from_url(url)

        if soup is None:
            print("Event not found. Skipping.")
            continue

        main_section = soup.find(id="main")
        if not main_section:
            print("No main section.")
            continue

        # collect section ids (id + title)
        sub_sections = get_sub_sections(main_section)

        # patterns for ID matching, this will cover long papers from main and findings
        patterns_id = [
            f"{year}{conf}-main",
            f"{year}findings-{conf}",
            f"{year}{conf}-long",
            f"{year}ijcnlp-main", #aacl
            f"{year}findings-ijcnlp",
            f"{year}ijcnlp-long",
            "tacl"
        ]

        patterns_title = ["long", "findings"]

        # ID-based matching
        target_sec_ids = [
            sid for sid, title in sub_sections
            if any(p in sid for p in patterns_id)
        ]

        # title fallback
        if not target_sec_ids:
            print("No section IDs matched: using title search.")
            target_sec_ids = [
                sid for sid, title in sub_sections
                if any(pt.lower() in title.lower() for pt in patterns_title)
            ]

        # final fallback (first section)
        # necessary to scrape early years with different proceedings structure
        if not target_sec_ids and sub_sections:
            target_sec_ids = [sub_sections[0][0]]

        print("Sections:", target_sec_ids)

        all_papers = []
        section_info = []

        for sec_id in target_sec_ids:

            papers = find_paper_list(soup, sec_id)
            print(f"Section {sec_id} papers: {len(papers)}")

            all_papers.extend(papers)
            section_info.append({
                "id": sec_id,
                "title": dict(sub_sections).get(sec_id, "Unknown")
            })

        total = len(all_papers)
        filtered = match_keywords(all_papers, keywords)
        matched = len(filtered)

        print(f"Total papers: {total}")
        print(f"Matched papers: {matched}")

        # save metadata JSON
        meta_dir = os.path.join(save_root, conf, str(year))
        os.makedirs(meta_dir, exist_ok=True)
        keyword_counts = count_keyword_hits(filtered, keywords)
        metadata = {
            "conference": conf,
            "year": year,
            "sections": section_info,
            "total_papers": total,
            "matched_papers": matched,
            "matched_keywords": keyword_counts
        }

        with open(os.path.join(meta_dir, "metadata.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # save matched pdfs
        for sec_id in target_sec_ids:
            papers = find_paper_list(soup, sec_id)
            filtered = match_keywords(papers, keywords)

            if not filtered:
                continue

            save_dir = os.path.join(save_root, conf, str(year), sec_id)
            save_papers(filtered, save_dir)
