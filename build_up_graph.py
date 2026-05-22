import os
import argparse
from tqdm import tqdm
from neo4j_client import Neo4jClient
from config import get_neo4j_config

# 导入普通实体 (批量化 UNWIND)
def import_entity(client, entity_type, entities):
    print(f'正在导入{entity_type}类数据 (批量加速)')
    # 使用参数化与批量解包，防注入且极速
    cypher = f"UNWIND $entities AS name CREATE (n:`{entity_type}` {{名称: name}})"
    client.run_query(cypher, entities=entities)

# 导入疾病类实体 (批量化 UNWIND)
def import_disease_data(client, entity_type, entities):
    print(f'正在导入{entity_type}类数据 (批量加速)')
    cypher = f"""
    UNWIND $entities AS disease 
    CREATE (n:`{entity_type}` {{
        名称: disease.名称, 
        疾病简介: disease.疾病简介, 
        疾病病因: disease.疾病病因, 
        预防措施: disease.预防措施, 
        治疗周期: disease.治疗周期, 
        治愈概率: disease.治愈概率, 
        疾病易感人群: disease.疾病易感人群
    }})
    """
    client.run_query(cypher, entities=entities)

# 导入关系 (批量化 UNWIND)
def create_all_relationship(client, all_relationship):
    print("正在导入关系 (批量加速).....")
    # 将关系元组转换为字典列表以便于 UNWIND
    rels = [{"t1": r[0], "n1": r[1], "rel": r[2], "t2": r[3], "n2": r[4]} for r in all_relationship]
    
    # 注意：Neo4j 的标签和关系类型不能直接参数化，需要我们在图谱类别已知（白名单）的情况下安全拼接
    # 由于这里是构建阶段，可以按关系类型分组批量导入
    rel_groups = {}
    for r in rels:
        key = (r['t1'], r['rel'], r['t2'])
        if key not in rel_groups:
            rel_groups[key] = []
        rel_groups[key].append({"n1": r['n1'], "n2": r['n2']})

    for (t1, rel_type, t2), pairs in tqdm(rel_groups.items()):
        cypher = f"""
        UNWIND $pairs AS pair
        MATCH (a:`{t1}` {{名称: pair.n1}}), (b:`{t2}` {{名称: pair.n2}})
        CREATE (a)-[r:`{rel_type}`]->(b)
        """
        client.run_query(cypher, pairs=pairs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过medical.json文件,创建一个知识图谱")
    neo4j_cfg = get_neo4j_config()
    parser.add_argument('--website', type=str, default=None, help='neo4j的bolt连接网站')
    parser.add_argument('--user', type=str, default=None, help='neo4j的用户名')
    parser.add_argument('--password', type=str, default=None, help='neo4j的密码')
    args = parser.parse_args()

    uri = args.website or neo4j_cfg["uri"]
    user = args.user or neo4j_cfg["user"]
    password = args.password or neo4j_cfg["password"]

    if not password:
        raise RuntimeError("Neo4j 密码未配置，请在 .streamlit/secrets.toml 或 --password 参数中提供")

    # 连接新版 Client
    client = Neo4jClient(uri, user, password)

    is_delete = input('注意:是否删除neo4j上的所有实体 (y/n):')
    if is_delete == 'y':
        client.run_query("MATCH (n) DETACH DELETE (n)")

    with open('./data/medical_new_2.json','r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    
    # 实体和关系解析逻辑保持不变 (省略中间提取逻辑，直接复用原代码的循环字典生成)
    # ... [此处保留原代码对 all_data 遍历生成 all_entity 和 relationship 的逻辑] ...
    
    # 由于上下文限制，中间解析数据的逻辑与原代码 100% 相同。
    # 假设 all_entity 和 relationship 已经构建完毕
    #所有实体
    all_entity = {
        "疾病": [],
        "药品": [],
        "食物": [],
        "检查项目":[],
        "科目":[],
        "疾病症状":[],
        "治疗方法":[],
        "药品商":[],
    }
    
    # 实体间的关系
    relationship = []
    for i,data in enumerate(all_data):
        if (len(data) < 3):
            continue
        data = eval(data[:-1])

        disease_name = data.get("name","")
        all_entity["疾病"].append({
            "名称":disease_name,
            "疾病简介": data.get("desc", ""),
            "疾病病因": data.get("cause", ""),
            "预防措施": data.get("prevent", ""),
            "治疗周期":data.get("cure_lasttime",""),
            "治愈概率": data.get("cured_prob", ""),
            "疾病易感人群": data.get("easy_get", ""),
        })

        drugs = data.get("common_drug", []) + data.get("recommand_drug", [])
        all_entity["药品"].extend(drugs)  # 添加药品实体
        if drugs:
            relationship.extend([("疾病", disease_name, "疾病使用药品", "药品",durg)for durg in drugs])

        do_eat = data.get("do_eat",[])+data.get("recommand_eat",[])
        no_eat = data.get("not_eat",[])
        all_entity["食物"].extend(do_eat+no_eat)
        if do_eat:
            relationship.extend([("疾病", disease_name,"疾病宜吃食物","食物",f) for f in do_eat])
        if no_eat:
            relationship.extend([("疾病", disease_name, "疾病忌吃食物", "食物", f) for f in no_eat])

        check = data.get("check", [])
        all_entity["检查项目"].extend(check)
        if check:
            relationship.extend([("疾病", disease_name, "疾病所需检查", "检查项目",ch) for ch in check])

        cure_department=data.get("cure_department", [])
        all_entity["科目"].extend(cure_department)
        if cure_department:
            relationship.append(("疾病", disease_name, "疾病所属科目", "科目",cure_department[-1]))

        symptom = data.get("symptom",[])
        for i,sy in enumerate(symptom):
            if symptom[i].endswith('...'):
                symptom[i] = symptom[i][:-3]
        all_entity["疾病症状"].extend(symptom)
        if symptom:
            relationship.extend([("疾病", disease_name, "疾病的症状", "疾病症状",sy )for sy in symptom])

        cure_way = data.get("cure_way", [])
        if cure_way:
            for i,cure_w in enumerate(cure_way):
                if(isinstance(cure_way[i], list)):
                    cure_way[i] = cure_way[i][0] #glm处理数据集偶尔有格式错误
            cure_way = [s for s in cure_way if len(s) >= 2]
            all_entity["治疗方法"].extend(cure_way)
            relationship.extend([("疾病", disease_name, "治疗的方法", "治疗方法", cure_w) for cure_w in cure_way])
            

        acompany_with = data.get("acompany", [])
        if acompany_with:
            relationship.extend([("疾病", disease_name, "疾病并发疾病", "疾病", disease) for disease in acompany_with])

        drug_detail = data.get("drug_detail",[])
        for detail in drug_detail:
            lis = detail.split(',')
            if(len(lis)!=2):
                continue
            p,d = lis[0],lis[1]
            all_entity["药品商"].append(d)
            all_entity["药品"].append(p)
            relationship.append(('药品商',d,"生产","药品",p))
    for i in range(len(relationship)):
        if len(relationship[i])!=5:
            print(relationship[i])
    relationship = list(set(relationship))
    all_entity = {k:(list(set(v)) if k!="疾病" else v)for k,v in all_entity.items()}
    
    # 保存关系 放到data下
    with open("./data/rel_aug.txt",'w',encoding='utf-8') as f:
        for rel in relationship:
            f.write(" ".join(rel))
            f.write('\n')

    if not os.path.exists('data/ent_aug'):
        os.mkdir('data/ent_aug')
    for k,v in all_entity.items():
        with open(f'data/ent_aug/{k}.txt','w',encoding='utf8') as f:
            if(k!='疾病'):
                for i,ent in enumerate(v):
                    f.write(ent+('\n' if i != len(v)-1 else ''))
            else:
                for i,ent in enumerate(v):
                    f.write(ent['名称']+('\n' if i != len(v)-1 else ''))
    
    for k in all_entity:
        if k != "疾病":
            import_entity(client, k, all_entity[k])
        else:
            import_disease_data(client, k, all_entity[k])
            
    create_all_relationship(client, relationship)
    client.close()