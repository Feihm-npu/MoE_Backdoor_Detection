import torch

def peek_pt(path):
    x = torch.load(path, map_location="cpu")
    print("top keys:", x.keys())
    print("meta keys:", x["meta"].keys())
    print("sample keys:", x["samples"][0].keys())
    # 看看 per_layer_hits 的形状
    l0 = x["samples"][0]["per_layer_hits"]["0"]
    print("layer0 hits len:", len(l0), "example:", l0[:8])

peek_pt("assets/01052026_detailed/routing_records_cleanft.pt")
