"""
构建检索索引：BM25 + 当前配置的远程 Embedding 模型 + FAISS 向量索引。

前置条件：先运行 parse_manuals.py 生成 data/ 下的 chunk 文件。

用法：
    conda run -n rag_agent python build_retrieval_index.py
    conda run -n rag_agent python build_retrieval_index.py --batch-size 64
"""

import argparse
import time

from retrieval_engine import RetrievalEngine


def main() -> None:
    """Parse CLI args and rebuild the BM25 + FAISS retrieval index."""
    parser = argparse.ArgumentParser(description="构建检索索引")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    args = parser.parse_args()

    engine = RetrievalEngine()

    t0 = time.time()
    engine.build_index(batch_size=args.batch_size)
    elapsed = time.time() - t0

    n_chunks = len(engine.retrieval_chunks)
    dim = engine.dense_vectors.shape[1] if engine.dense_vectors is not None else 0
    print(f"\n索引构建完成: {n_chunks} 检索块, 向量维度 {dim}, 耗时 {elapsed:.1f}s")
    print(f"  FAISS: {engine.faiss_path}")
    print(f"  元数据: {engine.metadata_path}")


if __name__ == "__main__":
    main()
