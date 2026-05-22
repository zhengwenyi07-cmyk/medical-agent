from neo4j import GraphDatabase

class Neo4jClient:
    """官方 Neo4j 驱动封装单例，提供连接池和防注入的参数化查询"""
    _instance = None

    def __new__(cls, uri, user, password):
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
            cls._instance.driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls._instance

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, cypher, **kwargs):
        """执行参数化 Cypher 查询，返回字典列表"""
        with self.driver.session() as session:
            result = session.run(cypher, **kwargs)
            return [record.data() for record in result]