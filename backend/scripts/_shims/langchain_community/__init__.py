"""项目内最小 shim：修复 ragas 对 langchain-community 已移除模块的强依赖。

ragas 0.3.x 在 ``ragas/llms/base.py`` 顶层无条件导入
``langchain_community.chat_models.vertexai.ChatVertexAI`` 与
``langchain_community.llms.VertexAI``，而 langchain-community 0.3+ 已移除
这两个模块。这两个类只出现在 ragas 的 LLM 工厂注册表里，我们在对照脚本中
注入自定义 wrapper，因此永远不会被实例化——用占位类即可满足导入。

仅当真实 langchain-community 缺失对应模块时，``compare_with_ragas.py``
才会把 ``scripts/_shims`` 注入 ``sys.path``。
"""
