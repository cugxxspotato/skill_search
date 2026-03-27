import os
import re
import shutil
import requests
import zipfile
import io
import time
import json
from bs4 import BeautifulSoup
from openai import OpenAI
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. 全局配置与 API 密钥
# ==========================================
GLM_API_KEY = "<KEY>"  
GITHUB_TOKEN = "<PASSWORD>" 
KAGGLE_TOKEN = "<PASSWORD>"
HF_TOKEN = "<PASSWORD>"

LINK_DIR = "./skill_link_files"
DATA_DIR = "./skill_datasets"   

if HF_TOKEN and len(HF_TOKEN) > 10:
    os.environ['HF_TOKEN'] = HF_TOKEN
    print("✅ HuggingFace Token 已配置，开启高速下载模式！")

client = OpenAI(
    api_key=GLM_API_KEY, 
    base_url="=====", 
    timeout=30.0 
)
MODEL_NAME = "glm-4.5-air"

KAGGLE_AVAILABLE = False
if KAGGLE_TOKEN and len(KAGGLE_TOKEN) > 10:
    os.environ['KAGGLE_API_TOKEN'] = KAGGLE_TOKEN
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        KAGGLE_AVAILABLE = True
        print("✅ Kaggle API 认证成功！")
    except Exception as e:
        print(f"⚠️ Kaggle 初始化失败: {e}")

# ==========================================
# 2. 多智能体协同 (Multi-Agent System) - 核心重构区
# ==========================================

def agent_analyst_create_rubric(skill_content):
    """【Agent 1: 分析师】取消 Ground Truth 限制，只关注高质量【输入数据】"""
    print("\n👨‍🔬 [Agent 分析师] 正在深度剖析 Skill，制定测试【输入数据】金标准...")
    prompt = f"""你是一位资深的 AI Agent 评测架构师。请阅读以下 Agent 技能描述，并为它设计一个【可操作输入数据 (Test Inputs/Fixtures) 的筛选标准】。
注意：我们在这个阶段**不需要**标准答案(Ground Truth)！只要数据能作为完美的测试输入源即可！

【技能描述】:\n{skill_content[:1500]}

你的任务是输出一份评估指南，包含：
1. **输入实体画像**：测试这个技能需要喂给它什么样的原始数据？(例如：一段待处理的文本、一个结构不完整的 CSV、一个需要检索的 JSON 库等)。
2. **数据完备度要求**：好的数据实体必须包含什么核心字段？
3. **打分维度**：定义 3 个打分维度 (如：领域匹配度、数据结构清晰度、Agent测试价值)。

请直接输出这份评估指南，它将作为后续裁判 Agent 的唯一评分标准。"""
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 分析师发生异常: {e}")
        return "通用标准：必须包含结构完整、领域相关的原始输入数据。"

def agent_strategist_generate_queries(skill_content, rubric):
    """【Agent 2: 战略家】彻底解决搜索死板问题，强制输出极简 1-2 个词"""
    print("🕵️ [Agent 战略家] 正在制定全网极简检索策略 (防止 API 搜索失败)...")
    prompt = f"""你是一个数据挖掘专家。请根据【Agent 技能】，提取用于在 Kaggle, HuggingFace, GitHub 搜索数据集的【核心实体词】。

【技能描述】:\n{skill_content[:1000]}

【严格红线警告】:
1. API 搜索非常死板！你输出的关键词必须是极度宽泛的 1~2 个纯英文名词 (例如: finance, genomics, weather, stock)。
2. 绝对、绝对不能包含 "dataset", "benchmark", "ground truth", "QA", "evaluation" 等冗长修饰词！否则会导致搜索失败！

请输出一个严格的 JSON 格式：
```json
{{
  "kaggle_query": "纯领域词",
  "huggingface_query": "纯领域词",
  "github_query": "纯领域词"
}}
```"""
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        content = res.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            print(f"   🎯 制定战术完毕! K:[{queries.get('kaggle_query')}] | HF:[{queries.get('huggingface_query')}] | GH:[{queries.get('github_query')}]")
            return queries
    except Exception as e:
        print(f"❌ 战略家发生异常: {e}")
    
    # 极简兜底策略
    return {"kaggle_query": "data", "huggingface_query": "data", "github_query": "data"}

# ==========================================
# 3. 强力爬虫模块 (带断线重连重试机制)
# ==========================================

def robust_retry(func, retries=3, delay=3):
    """不死鸟重试装饰器/执行器"""
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"   ⚠️ 下载波动 (尝试 {attempt+1}/{retries}): {e}")
                time.sleep(delay)
        print("   ❌ 重试耗尽，下载失败。")
        return False
    return wrapper

def _download_hf_file(repo_id, filename, target_dir):
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=target_dir)
    return True

robust_hf_download = robust_retry(_download_hf_file)
robust_requests_get = robust_retry(requests.get)

def download_from_huggingface(query, target_dir, max_count=3):
    print(f"\n🔍 [HuggingFace] 执行极简搜索: '{query}'")
    api = HfApi()
    try:
        datasets = list(api.list_datasets(search=query, limit=20))
        count = 0
        for dataset in datasets:
            if count >= max_count: break
            files = api.list_repo_files(dataset.id, repo_type="dataset")
            valid_files = [f for f in files if f.endswith(('.json', '.jsonl', '.csv', '.parquet'))][:3]
            if valid_files:
                repo_dir = os.path.join(target_dir, f"hf_{count+1}")
                os.makedirs(repo_dir, exist_ok=True)
                print(f"📥 [HF] 下载匹配项目 ({count+1}/{max_count}): {dataset.id}...")
                for file in valid_files:
                    robust_hf_download(dataset.id, file, repo_dir)
                count += 1
        return count
    except Exception as e:
        print(f"⚠️ HuggingFace 下载发生异常: {e}")
        return 0

def download_from_github(query, target_dir, max_count=3):
    print(f"\n🔍 [GitHub] 执行极简搜索: '{query}'")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    search_url = f"https://api.github.com/search/repositories?q={query} in:readme,description&sort=stars&per_page=15"
    try:
        response = robust_requests_get(search_url, headers=headers)
        if response and response.status_code == 200 and response.json().get("items"):
            count = 0
            for repo in response.json()["items"]:
                if count >= max_count: break
                repo_dir = os.path.join(target_dir, f"github_{count+1}")
                os.makedirs(repo_dir, exist_ok=True)
                zip_url = f"https://api.github.com/repos/{repo['full_name']}/zipball/{repo['default_branch']}"
                print(f"📥 [GitHub] 下载匹配仓库 ({count+1}/{max_count}): {repo['full_name']}...")
                zip_resp = robust_requests_get(zip_url, headers=headers)
                if zip_resp and zip_resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zip_ref:
                        zip_ref.extractall(repo_dir)
                    count += 1
            return count
    except: return 0

def download_from_kaggle(query, target_dir, max_count=3):
    if not KAGGLE_AVAILABLE: return 0
    print(f"\n🔍 [Kaggle] 执行极简搜索: '{query}'")
    try:
        datasets = kaggle_api.dataset_list(search=query)
        count = 0
        for dataset in datasets:
            if count >= max_count: break
            repo_dir = os.path.join(target_dir, f"kaggle_{count+1}")
            os.makedirs(repo_dir, exist_ok=True)
            print(f"📥 [Kaggle] 下载匹配项目 ({count+1}/{max_count}): {dataset.ref}...")
            
            for attempt in range(3):
                try:
                    kaggle_api.dataset_download_files(dataset.ref, path=repo_dir, unzip=True)
                    break
                except Exception:
                    time.sleep(3)
            count += 1
        return count
    except: return 0

# ==========================================
# 4. 铁面裁判模块 (严格打分制)
# ==========================================

def get_project_summary(project_dir):
    def safe_read(p, max_c=600):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f: return f.read(max_c)
        except: return ""
        
    readme_text, data_files = "", []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            file_lower = file.lower()
            path = os.path.join(root, file)
            if file_lower in ['readme.md', 'readme.txt'] and not readme_text:
                readme_text = safe_read(path, 800)
            elif file_lower.endswith(('.csv', '.json', '.jsonl', '.parquet', '.txt')) and file_lower not in ['package.json', 'requirements.txt']:
                data_files.append(os.path.relpath(path, project_dir))
                
    if not readme_text and data_files:
        readme_text = f"[无 README，抽取数据样本]:\n{safe_read(os.path.join(project_dir, data_files[0]), 500)}"
    return readme_text, data_files

def agent_judge_score_projects(rubric, target_dir):
    """【Agent 3: 铁面裁判】移除 Ground Truth 要求，专注输入数据的质量打分"""
    print(f"\n⚖️ [Agent 裁判] 正在根据金标准，对所有下载实体进行量化打分...")
    
    projects = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    valid_projects = {}
    context = ""
    
    for p in projects:
        p_dir = os.path.join(target_dir, p)
        readme, dfiles = get_project_summary(p_dir)
        if dfiles:
            valid_projects[p] = dfiles
            dfiles_str = ", ".join(dfiles[:5])
            context += f"\n=== 项目 ID: {p} ===\n【描述/样本】:\n{readme}\n【包含数据】: {dfiles_str}\n"

    if not valid_projects:
        print("⚠️ 均未找到有效数据文件。")
        shutil.rmtree(target_dir) 
        return

    prompt = f"""你是一个苛刻的 AI 评测数据集裁判。请根据【筛选金标准】，对以下【候选项目】进行 0-100 分的量化打分。
【打分原则红线】：
1. 我们**绝对不需要**项目自带标准答案 (Ground Truth)！
2. 只要它是一个优质的、结构清晰的【测试输入源】(例如包含待处理业务数据的 CSV/JSON)，就应该给高分 (>80分)。
3. 只有那些纯软件代码库(毫无业务数据)的项目，才给低分 (<30分)。

【筛选金标准】:
{rubric}

【候选项目汇总】:
{context}

请严格输出一段 JSON 格式的数据。
格式范例：
{{
  "results": [
    {{"id": "kaggle_1", "score": 90, "reason": "包含完整业务结构数据，非常适合作为测试 Agent 的输入源"}},
    {{"id": "github_2", "score": 20, "reason": "只是一个空的代码库，缺乏测试输入数据"}}
  ]
}}"""
    
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        json_str = re.search(r'\{.*\}', res.choices[0].message.content, re.DOTALL).group(0)
        eval_data = json.loads(json_str)
        
        scores = eval_data.get("results", [])
        scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        print("\n📊 --- 最终成绩单 ---")
        best_project_ids = []
        for s in scores:
            p_id = s.get("id")
            score = s.get("score", 0)
            reason = s.get("reason", "")
            print(f"[{score} 分] 项目: {p_id} | 理由: {reason}")
            # 录取及格分数 (>50)，且属于真实爬到的项目，最多 5 个
            if p_id in valid_projects and score >= 50 and len(best_project_ids) < 5:
                best_project_ids.append(p_id)
                
    except Exception as e:
        print(f"❌ 裁判打分环节发生格式异常: {e}")
        return

    if not best_project_ids:
        print("❌ 裁判判定：所有数据源均为无用的纯代码库，不合格，全部淘汰。")
        shutil.rmtree(target_dir)
        return
        
    print(f"\n🧹 清理阶段：淘汰低分项，提取 {len(best_project_ids)} 个优质输入源实体...")
    safe_temp_dir = os.path.join(DATA_DIR, f"TEMP_{os.path.basename(target_dir)}")
    os.makedirs(safe_temp_dir, exist_ok=True)
    
    saved_count = 0
    for p_id in best_project_ids:
        p_dir = os.path.join(target_dir, p_id)
        for rel_file in valid_projects[p_id]:
            abs_file = os.path.join(p_dir, rel_file)
            new_name = f"TOP_{p_id}_{os.path.basename(abs_file)}"
            shutil.copy(abs_file, os.path.join(safe_temp_dir, new_name))
            saved_count += 1
            
    shutil.rmtree(target_dir) 
    os.rename(safe_temp_dir, target_dir) 
    print(f"🎉 成功！已提取出 {saved_count} 个完美的测试输入文件，保存至: {target_dir}")

# ==========================================
# 5. 基础解析代码 & 主干
# ==========================================
def fetch_skills_from_url(url):
    url = url.strip()
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    if "clawhub.ai" in url:
        url = f"https://wry-manatee-359.convex.site/api/v1/download?slug={url.split('/')[-1].split('?')[0]}"
    elif "github.com" in url:
        if "/blob/" in url: url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            parts = url.rstrip('/').split('/')
            if len(parts) >= 5: url = f"https://api.github.com/repos/{parts[3]}/{parts[4]}/zipball"
    try:
        resp = robust_requests_get(url, headers=headers)
        if not resp or resp.status_code != 200: return []
        ct = resp.headers.get('Content-Type', '')
        if 'zip' in ct or 'zipball' in url or 'slug=' in url:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                t = ['skill.md', 'skills.md', 'instruction.md']
                s_fs = [f for f in z.namelist() if os.path.basename(f).lower() in t]
                if not s_fs:
                    r_fs = sorted([f for f in z.namelist() if os.path.basename(f).lower() == 'readme.md'], key=lambda x: len(x.split('/')))
                    if r_fs: s_fs = [r_fs[0]]
                if not s_fs: return []
                return [z.read(sf).decode('utf-8', errors='ignore') for sf in s_fs]
        if 'text/html' in ct:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(["script", "style", "nav"]): tag.extract()
            return [re.sub(r'\n\s*\n', '\n\n', soup.get_text(separator='\n'))[:5000]] 
        return [resp.text]
    except: return []

def get_skill_name_with_llm(skill_content):
    prompt = f"请根据以下 Agent 技能内容，生成极简短英文名（仅小写和下划线）。不要废话：\n{skill_content[:800]}"
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        if hasattr(res, 'choices') and res.choices:
            clean_name = re.sub(r'[^a-z0-9_]', '', res.choices[0].message.content.strip().lower().replace(' ', '_'))
            if clean_name: return clean_name[:20] 
    except: pass
    return "unknown_skill"

print("🚀 启动 Agent Skill 工业级测试源挖掘流水线...")
os.makedirs(LINK_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

txt_files = [f for f in os.listdir(LINK_DIR) if f.endswith('.txt')]
if not txt_files: print(f"⚠️ {LINK_DIR} 下没找到 txt。")
else:
    for txt_name in txt_files:
        with open(os.path.join(LINK_DIR, txt_name), 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f if line.strip()]
        for url in links:
            print(f"\n{'='*70}\n🔗 目标载入: {url}")
            skill_texts = fetch_skills_from_url(url)
            if not skill_texts: continue
            
            for idx, skill_text in enumerate(skill_texts):
                print(f"\n📌 --- 任务 {idx+1}/{len(skill_texts)} ---")
                
                skill_name = get_skill_name_with_llm(skill_text)
                if len(skill_texts) > 1: skill_name = f"{skill_name}_p{idx+1}"
                print(f"🏷️ 目标代号: {skill_name}")
                
                # 核心机制启动
                rubric = agent_analyst_create_rubric(skill_text)
                queries = agent_strategist_generate_queries(skill_text, rubric)
                
                target_dir = os.path.join(DATA_DIR, skill_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # 请求三大平台
                download_from_kaggle(queries.get("kaggle_query", "data"), target_dir, 3)
                download_from_huggingface(queries.get("huggingface_query", "data"), target_dir, 3)
                download_from_github(queries.get("github_query", "data"), target_dir, 3)
                
                if os.listdir(target_dir):
                    agent_judge_score_projects(rubric, target_dir)
                else:
                    print(f"⚠️ 未能下载到任何数据。")

print("\n🏁 任务彻底完毕！")