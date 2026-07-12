import os
import re
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

LOGGER = logging.getLogger("plumetrace.rag_engine")

# Basic stop words to filter out before computing TF-IDF
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", 
    "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", 
    "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", 
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", 
    "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", 
    "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", 
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", 
    "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", 
    "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", 
    "that", "thats", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", 
    "this", "those", "through", "to", "too", "under", "until", "up", "very", 
    "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", "what", 
    "whats", "when", "whens", "where", "wheres", "which", "while", "who", 
    "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", 
    "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

class SimpleVectorStore:
    """A lightweight, self-contained vector store using TF-IDF and Cosine Similarity."""
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[List[float]] = []

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Docs is a list of dicts with 'text' and 'metadata' keys."""
        self.documents.extend(docs)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize words, lowercased, filtering out short words and stop words
        words = re.findall(r'\b[a-zA-Z]{3,20}\b', text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def _build_index(self):
        if not self.documents:
            return
        
        doc_tfs = []
        vocab = set()
        for doc in self.documents:
            tokens = self._tokenize(doc['text'])
            tf: Dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0.0) + 1.0
            doc_tfs.append(tf)
            vocab.update(tf.keys())
            
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
        
        num_docs = len(self.documents)
        doc_counts: Dict[str, int] = {}
        for tf in doc_tfs:
            for word in tf.keys():
                doc_counts[word] = doc_counts.get(word, 0) + 1
                
        self.idf = {}
        for word, count in doc_counts.items():
            # Standard IDF formula with smoothing
            self.idf[word] = math.log(1.0 + (num_docs / count))

        self.vectors = []
        for tf in doc_tfs:
            vec = [0.0] * len(self.vocabulary)
            for word, val in tf.items():
                idx = self.vocabulary[word]
                vec[idx] = val * self.idf[word]
            self.vectors.append(self._normalize(vec))

    def _normalize(self, vec: List[float]) -> List[float]:
        sq_sum = sum(x*x for x in vec)
        if sq_sum == 0.0:
            return vec
        norm = math.sqrt(sq_sum)
        return [x / norm for x in vec]

    def similarity_search(self, query: str, k: int = 2) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents or not self.vocabulary:
            return []
        
        tokens = self._tokenize(query)
        q_vec = [0.0] * len(self.vocabulary)
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                q_vec[idx] += 1.0 * self.idf[token]
        q_vec = self._normalize(q_vec)
        
        results = []
        for i, doc_vec in enumerate(self.vectors):
            # Cosine similarity (dot product of normalized vectors)
            score = sum(a*b for a, b in zip(q_vec, doc_vec))
            if score > 0.0:
                results.append((self.documents[i], score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# Global RAG store instance
RAG_STORE = SimpleVectorStore()

def initialize_rag(policy_dir_path: str = None) -> int:
    """Load policy documents, parse them into chunks, and index them in RAG_STORE."""
    if policy_dir_path is None:
        # Resolve path relative to backend root
        project_root = Path(__file__).resolve().parents[2]
        policy_dir_path = str(project_root / "backend" / "policy_documents")

    if not os.path.exists(policy_dir_path):
        LOGGER.warning("Policy directory not found: %s", policy_dir_path)
        return 0

    chunks = []
    for filename in os.listdir(policy_dir_path):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(policy_dir_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Custom chunk parsing: if file contains distinct profiles, split by profile headers
        if "Profile -" in content:
            # Split by headers (e.g. [Facility Infractions Profile - ...])
            profiles = re.split(r'(?=\[Facility Infractions Profile -)', content)
            for profile in profiles:
                profile_clean = profile.strip()
                if profile_clean:
                    # Extract profile title
                    title_match = re.search(r'\[([^\]]+)\]', profile_clean)
                    title = title_match.group(1) if title_match else "Infraction Profile"
                    chunks.append({
                        "text": profile_clean,
                        "metadata": {"source": filename, "type": "infraction_history", "title": title}
                    })
        else:
            # For general policies, split by major sections/paragraphs
            paragraphs = content.split("\n\n")
            for idx, p in enumerate(paragraphs):
                p_clean = p.strip()
                if p_clean:
                    chunks.append({
                        "text": p_clean,
                        "metadata": {"source": filename, "type": "policy_guideline", "chunk_index": idx}
                    })

    RAG_STORE.add_documents(chunks)
    LOGGER.info("Successfully loaded and TF-IDF indexed %d policy document chunks.", len(chunks))
    return len(chunks)


def query_rag(query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Query the TF-IDF RAG engine and return formatted snippet dictionaries."""
    results = RAG_STORE.similarity_search(query_text, k=top_k)
    return [
        {
            "text": doc["text"],
            "source": doc["metadata"].get("source", "unknown"),
            "type": doc["metadata"].get("type", "unknown"),
            "score": round(score, 3)
        }
        for doc, score in results
    ]
