from faiss_manager import FaissManager
from storage_provider import FileSystemStorageProvider
from index_builder import IndexBuilder

json_file = "data/marques-francaises-latest-50k.json"
index_path = "data/marques_index.faiss"
mapping_path = "data/marques_mapping.json"

storage = FileSystemStorageProvider(index_path, mapping_path)
faiss_manager = FaissManager(storage_provider=storage)
indexBuilder = IndexBuilder(faiss_manager)

data = storage.load_data(json_file)

marks = []
for record in data:
    try:
        marks.append(record["Mark"])
    except:
        print(record["ApplicationNumber"], record["ApplicationDate"])
        pass

indexBuilder.build_index_from_texts(marks)