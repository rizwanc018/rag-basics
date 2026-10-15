import lancedb

db = lancedb.connect("db/lancedb")
table = db.open_table("docling")
# DocLayNet

result = table.search(query="DocLayNet", query_type="vector").limit(3)
result.to_pandas()


for r in result.to_list():
    print(r["text"])
