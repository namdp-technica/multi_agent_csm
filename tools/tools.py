import requests
import base64
import os
from typing import Dict, Any
from utils.helper_workflow import load_config
CONFIG_PATH = os.path.join(os.path.dirname(__file__),"..", "config", "config.yaml")
config = load_config(config_path=CONFIG_PATH)

def agent_tool(func):
    return func
API_URL = config["milvus"]["search_url"]
class Api:
    
    def __init__(self, output_folder: str = config["paths"]["tools_results"]):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
    
    @agent_tool
    def image_search(self, query: str, k: int = config["milvus"]["default_top_k"]) -> Dict[str, Any]:
        try:
            payload = {"query": query, "top": k}
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                return {"query": query, "total_found": 0, "images": [], "error": f"API error: {response.text}"}
            data = response.json()
            results = data.get("results", [])
            images = []
            for res in results:
                doc_id = res.get("doc_id")
                score = res.get("score")
                filepath_server = res.get("filepath")
                image_base64 = res.get("image_base64")
                local_filename = f"doc_{doc_id}.png"
                local_path = os.path.join(self.output_folder, local_filename)
                if image_base64:
                    image_clean = image_base64.split(",")[1] if "," in image_base64 else image_base64
                    with open(local_path, "wb") as f:
                        f.write(base64.b64decode(image_clean))
                images.append({
                    "id": f"doc_{doc_id}",
                    "path": local_path,
                    "url": None,
                    "description": f"Image from doc_id {doc_id} related to: {query}",
                    "relevance_score": score,
                    "metadata": {
                        "original_server_path": filepath_server,
                        "local_filename": local_filename,
                        "format": "png"
                    }
                })
            return {
                "query": query,
                "total_found": len(images),
                "images": images,
                "search_metadata": {
                    "search_time": "API call",
                    "algorithm": "milvus_search_default_base64"
                }
            }
        except Exception as e:
            return {"query": query, "total_found": 0, "images": [], "error": f"Image search failed: {str(e)}"}

if __name__ == "__main__":
    print(":test_tube: Testing Image Search Tool with Milvus API...")
    tool = Api(output_folder=config["paths"]["tools_results"])
    query = "温度差荷重の記号"
    k = 3
    results = tool.image_search(query=query, k=k)
    print("\n=== Search Result ===")
    print(f"Query: {results['query']}")
    print(f"Total Found: {results['total_found']}")
    for img in results["images"]:
        print(f"- ID: {img['id']}")
        print(f"  Path: {img['path']}")
        print(f"  Score: {img['relevance_score']}")
        print(f"  Original Server Path: {img['metadata']['original_server_path']}")